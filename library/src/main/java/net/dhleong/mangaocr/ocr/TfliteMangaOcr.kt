package net.dhleong.mangaocr.ocr

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import net.dhleong.mangaocr.MangaOcr
import net.dhleong.mangaocr.Vocab
import net.dhleong.mangaocr.hub.HfHubRepo
import net.dhleong.mangaocr.tflite.TfLiteHelper
import org.tensorflow.lite.DataType
import org.tensorflow.lite.InterpreterApi
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.nio.FloatBuffer
import java.nio.LongBuffer

class TfliteMangaOcr private constructor(
    private val interpreter: InterpreterApi,
    private val vocab: Vocab,
//    private val processor: ImageProcessor<TensorImage> =
//        ImageProcessor { floats, shape ->
//            TensorImage.
//        },
) : MangaOcr {
    private val imageProcessor =
        ImageProcessor
            .Builder()
            .add(ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
            .add(
                NormalizeOp(
                    floatArrayOf(0f, 0f, 0f),
                    floatArrayOf(1f, 1f, 1f),
                ),
            ).build()

    override suspend fun process(bitmap: Bitmap): Flow<MangaOcr.Result> =
        flow {
            val inputImage =
                TensorImage(DataType.FLOAT32).apply {
                    load(bitmap)
                }
            val image = imageProcessor.process(inputImage)

            val tokenIds = LongBuffer.allocate(80)
            tokenIds.put(2)

            val logitsIdx = 2 // I promise
            val outputTokens = FloatBuffer.allocate(vocab.size)

            for (i in 0 until interpreter.inputTensorCount) {
                println("@$i: ${interpreter.getInputTensor(i).name()}")
            }

            println("inputs=${image.buffer} / ${tokenIds.array()}")
            for (tokensCount in 1 until 80) {
                tokenIds.flip()
                val outputs = mutableMapOf<Int, Any>(logitsIdx to outputTokens)
                interpreter.runForMultipleInputsOutputs(
                    arrayOf(
                        image.buffer,
                        tokenIds,
                    ),
                    outputs,
                )

                val maxTokenId = outputTokens.findMaxArg(0, vocab.size)
                if (maxTokenId == 3) {
                    println("Found end!")
                    break
                }

                val token = vocab.lookupToken(maxTokenId)
                println("Max tokenId (${outputTokens.remaining()}) = ($maxTokenId) $token")
                emit(MangaOcr.Result.Partial(token))

                tokenIds.limit(tokensCount + 1)
                tokenIds.position(tokensCount)
                tokenIds.put(maxTokenId.toLong())

//            println("outputs=$outputs / ${outputs.size} in $run ms")
//            for (i in 0 until interpreter.outputTensorCount) {
//                val output = interpreter.getOutputTensor(i)
//                println("output[$i].shape() == ${output.shape().toList()}")
//                println("output[$i].numElements() == ${output.numElements()}")
//            }
//            println(vocab.size)
//            println(outputTokens.array().toList())
            }

            emit(MangaOcr.Result.FinalResult("TODO"))
        }.flowOn(Dispatchers.IO)

    companion object {
        suspend fun initialize(context: Context): TfliteMangaOcr =
            coroutineScope {
                val start = System.currentTimeMillis()

                val ctx = context.applicationContext
                val tflite = async { TfLiteHelper.initialize(ctx) }
                val interpreter =
                    async {
                        val modelPath =
                            HfHubRepo("dhleong/manga-ocr-android").resolveLocalPath(
                                ctx,
                                "mangaocr.tflite",
                                sha256 = "621ab13f09751843dc39c4ba608066e60c3db285237bffa52107b183cda72c0b",
                            )

                        Log.v(
                            "Tflite",
                            "Prepared model file in ${System.currentTimeMillis() - start} ms",
                        )
                        tflite.await()
                        Log.v("Tflite", "Runtime ready")

                        InterpreterApi.create(
                            modelPath,
                            InterpreterApi.Options().apply {
                                runtime = InterpreterApi.Options.TfLiteRuntime.FROM_SYSTEM_ONLY
                            },
                        )
                    }
                val vocab = async { Vocab.load(context) }

                TfliteMangaOcr(interpreter.await(), vocab.await()).also {
                    Log.v("Tflite", "Session ready in ${System.currentTimeMillis() - start} ms")
                }
            }
    }
}
