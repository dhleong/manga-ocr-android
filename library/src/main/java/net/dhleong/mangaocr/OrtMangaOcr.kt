package net.dhleong.mangaocr

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OnnxTensorLike
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.withContext
import net.dhleong.mangaocr.hub.HfHubRepo
import java.io.File
import java.nio.LongBuffer

class OrtMangaOcr private constructor(
    private val session: OrtSession,
    private val vocab: Vocab,
    private val processor: ImageProcessor<OnnxTensorLike> =
        ImageProcessor { floats, shape ->
            OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), floats, shape)
        },
) : MangaOcr {
    override suspend fun process(bitmap: Bitmap) {
        // TODO: Cancel any pending invocation?
        val maxChars = 80
        val image = processor.preprocess(bitmap)
        val tokenIds = LongBuffer.allocate(maxChars)
        tokenIds.put(2) // start token

        for (tokensCount in 1 until maxChars) {
            // Prepare to read the tokenIds:
            tokenIds.flip()

            val tokenIdsTensor =
                OnnxTensor.createTensor(
                    OrtEnvironment.getEnvironment(),
                    tokenIds,
                    longArrayOf(1, tokensCount.toLong()),
                )
            val outputs = session.run(mapOf("image" to image, "token_ids" to tokenIdsTensor))
            val logitsTensor = outputs.get("logits").get() as OnnxTensor

            val logits = logitsTensor.floatBuffer
            val shape = logitsTensor.info.shape
            if (shape.size != 3 || shape[0] != 1L) {
                throw IllegalStateException("Unexpected output tensor shape: ${shape.toList()}")
            }
            if (logits.limit() == 0) {
                throw IllegalStateException("Empty logits")
            }
            val resultRows = shape[1].toInt()
            val count = shape[2].toInt()
            if (resultRows < 1) {
                throw IllegalStateException("No resultRows")
            }
            Log.v("ORT", "Result rows = $resultRows")
            val start = count * (resultRows - 1)
            val end = start + count

            // find argmax
            var maxTokenId = -1
            var maxArg = 0f
            for (i in start until end) {
                val value = logits.get(i)
                if (maxTokenId < 0 || value > maxArg) {
//                    Log.v("ORT", "max @$i token $i = $value")
                    maxTokenId = (i - start)
//                    maxTokenId = candidateTokenId
                    maxArg = value
                }
//                candidateTokenId += 1
            }

            if (maxTokenId < 0) {
                throw IllegalStateException("No max token found")
            }

            tokenIds.limit(tokensCount + 1)
            tokenIds.position(tokensCount)
            tokenIds.put(maxTokenId.toLong())
            val token = vocab.lookupToken(maxTokenId)
            Log.v("ORT", "Got token $maxTokenId ($token); tokenIds=${tokenIds.array().take(tokensCount).toList()}")

            // Quit on end token
            if (maxTokenId == 3) {
                break
            }

            // Quick reject, mainly for testing:
            if (tokenIds.limit() >= 5 && tokenIds.array().take(5).all { it == 2L }) {
                Log.v("ORT", "Doesn't look like anything to me")
                break
            }
        }

        tokenIds.flip()

        val result = mutableListOf<String>()
        while (tokenIds.hasRemaining()) {
            val tokenId = tokenIds.get()
            if (tokenId >= 5) {
                result += vocab.lookupToken(tokenId.toInt())
            }
        }
        Log.v("ORT", "Got: $result")
    }

    companion object {
        suspend fun initialize(context: Context): OrtMangaOcr =
            coroutineScope {
                val start = System.currentTimeMillis()

                val ctx = context.applicationContext
                val repo = HfHubRepo("mayocream/koharu")
                val session =
                    async {
                        val modelPath =
                            repo.resolveLocalPath(
                                ctx,
                                "manga-ocr.onnx",
                                sha256 = "ac11c392af15df90f07d9a6473c737b7c18fdd31af6756768b6a7886f1fb3be1",
                            )
                        Log.v("ORT", "Prepared model file in ${System.currentTimeMillis() - start} ms")
                        buildSession(modelPath)
                    }
                val vocab =
                    async {
                        val vocabPath =
                            repo.resolveLocalPath(
                                ctx,
                                "vocab.txt",
                                sha256 = "344fbb6b8bf18c57839e924e2c9365434697e0227fac00b88bb4899b78aa594d",
                            )
                        Log.v("ORT", "Prepared vocab file in ${System.currentTimeMillis() - start} ms")
                        Vocab.loadFromFile(vocabPath)
                    }

                OrtMangaOcr(session.await(), vocab.await()).also {
                    Log.v("ORT", "Session ready in ${System.currentTimeMillis() - start} ms")
                }
            }

        private suspend fun buildSession(modelPath: File): OrtSession =
            withContext(Dispatchers.IO) {
                OrtEnvironment.getEnvironment().createSession(
                    modelPath.absolutePath,
                    OrtSession.SessionOptions().apply {
                        setOptimizationLevel(OrtSession.SessionOptions.OptLevel.EXTENDED_OPT)
//                            setSessionLogLevel(OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE)
                        setIntraOpNumThreads(4)
                    },
                )
            }
    }
}
