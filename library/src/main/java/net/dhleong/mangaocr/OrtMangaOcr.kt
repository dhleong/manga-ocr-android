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
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import net.dhleong.mangaocr.hub.HfHubRepo
import net.dhleong.mangaocr.onnx.FloatTensor
import net.dhleong.mangaocr.onnx.createSession
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
    override suspend fun process(bitmap: Bitmap): Flow<MangaOcr.Result> =
        flow {
            // TODO: Cancel any pending invocation?
            val maxChars = 80
            val image = processor.preprocess(bitmap)
            val tokenIds = LongBuffer.allocate(maxChars)
            tokenIds.put(2) // start token

            val result = StringBuilder(tokenIds.limit())

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
                val logitsTensor = FloatTensor.from(outputs.get("logits").get())
                if (logitsTensor.rowsCount == 0) {
                    throw IllegalStateException("No resultRows")
                }

                val maxTokenId = logitsTensor.maxValuedIndexInRow(logitsTensor.lastRowIndex)
                tokenIds.limit(tokensCount + 1)
                tokenIds.position(tokensCount)
                tokenIds.put(maxTokenId.toLong())

                val token = vocab.lookupToken(maxTokenId)
                if (maxTokenId >= 5) {
                    result.append(token)
                    emit(MangaOcr.Result.Partial(result))
                }

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

            emit(MangaOcr.Result.FinalResult(result.toString()))
            Log.v("ORT", "Got: $result")
        }.flowOn(Dispatchers.IO)

    companion object {
        suspend fun initialize(context: Context): OrtMangaOcr =
            coroutineScope {
                val start = System.currentTimeMillis()

                val ctx = context.applicationContext
                val session =
                    async {
                        val modelPath =
                            HfHubRepo("dhleong/manga-ocr-android").resolveLocalPath(
                                ctx,
                                "manga-ocr.quant.onnx",
                                sha256 = "73ee2e80cdce8f47590cb84486947f3bf0c1587bf46addb9007a7f7469ee332e",
//                                "manga-ocr.preprocessed.static.onnx",
//                                sha256 = "32c3dafff843ec110de0fe6ed8c1ffee848965bc6cd7439cbf0bd653f7eef4c0",
                            )
                        Log.v("ORT", "Prepared model file in ${System.currentTimeMillis() - start} ms")
                        buildSession(modelPath)
                    }
                val vocab = async { Vocab.fetch(context) }

                OrtMangaOcr(session.await(), vocab.await()).also {
                    Log.v("ORT", "Session ready in ${System.currentTimeMillis() - start} ms")
                }
            }

        private suspend fun buildSession(modelPath: File): OrtSession =
            createSession(modelPath) {
                setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)

//                addXnnpack(emptyMap())
//                addNnapi()
                addCPU(true)
                Log.v("ORT", "procs=${Runtime.getRuntime().availableProcessors()}")
                setIntraOpNumThreads(Runtime.getRuntime().availableProcessors().coerceAtMost(4))

                // This is the recommended xnnpack config, but it's way slower:
//                        addXnnpack(mapOf("intra_op_num_threads" to "4"))
//                        addConfigEntry("session.intra_op.allow_spinning", "0")
//                        setIntraOpNumThreads(1)
            }
    }
}
