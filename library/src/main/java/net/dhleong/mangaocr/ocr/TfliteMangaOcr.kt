package net.dhleong.mangaocr.ocr

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.android.gms.tflite.java.TfLite
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import net.dhleong.mangaocr.MangaOcr
import net.dhleong.mangaocr.hub.HfHubRepo
import net.dhleong.mangaocr.onnx.FloatTensor.Companion.allocateFloatOutputTensor
import net.dhleong.mangaocr.tflite.await
import org.tensorflow.lite.InterpreterApi
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.nio.LongBuffer
import kotlin.system.measureTimeMillis

class TfliteMangaOcr(
    private val encoder: InterpreterApi,
    private val decoder: InterpreterApi,
    private val maxChars: Int = 300,
    private val targetWidth: Int = 224,
    private val targetHeight: Int = 224,
) : MangaOcr {
    override suspend fun process(bitmap: Bitmap): Flow<MangaOcr.Result> =
        flow {
            val imageProcessor =
                ImageProcessor
                    .Builder()
                    .add(ResizeOp(targetHeight, targetWidth, ResizeOp.ResizeMethod.BILINEAR))
                    .add(NormalizeOp(127.5f, 127.5f)) // [ -1, 1 ]
                    .build()
            val processed = imageProcessor.process(TensorImage.fromBitmap(bitmap))

            for (i in 0 until encoder.outputTensorCount) {
                val t = encoder.getOutputTensor(i)
                Log.v("TfliteDetector", "encoder($i): ${t.name()} / ${t.shape().toList()}")
            }

            val encoded = encoder.allocateFloatOutputTensor(0)
            Log.v("TfliteDetector", "encode(...): ${encoded.buffer.array().toList()}")
            val encodeTimeMs =
                measureTimeMillis {
                    encoder.run(processed.buffer, encoded.buffer)
                }
            Log.v("TfliteDetector", "encode($encodeTimeMs ms): ${encoded.buffer.array().toList()}")

            // TODO: Loop with encoded into the decoder, accumulating tokens
            val tokenIds = LongBuffer.allocate(maxChars)
            tokenIds.put(2) // start token

            // TODO: Get output buffer
            Log.v("TfliteMangaOcr", "output=${decoder.getOutputTensor(0).name()}")
            decoder.runForMultipleInputsOutputs(arrayOf(encoded.buffer, tokenIds), emptyMap())

            // TODO:
            emit(MangaOcr.Result.FinalResult(""))
        }.flowOn(Dispatchers.IO)

    companion object {
        data class ModelPath(
            val path: String,
            val sha256: String,
        )

        private val MODEL_ENCODER =
            ModelPath(
                path = "manga-ocr.encoder.tflite",
                sha256 = "",
            )

        private val MODEL_DECODER =
            ModelPath(
                path = "manga-ocr.decoder.tflite",
                sha256 = "",
            )

        suspend fun initialize(context: Context): MangaOcr =
            coroutineScope {
                val encoderFile =
                    async {
                        HfHubRepo("dhleong/manga-ocr-android").resolveLocalPath(
                            context,
                            MODEL_ENCODER.path,
                            sha256 = MODEL_ENCODER.sha256,
                        )
                    }

                val decoderFile =
                    async {
                        HfHubRepo("dhleong/manga-ocr-android").resolveLocalPath(
                            context,
                            MODEL_DECODER.path,
                            sha256 = MODEL_DECODER.sha256,
                        )
                    }
                val initialized =
                    async {
                        TfLite.initialize(context).await()
                    }

                initialized.await()

                val encoder =
                    InterpreterApi.create(
                        encoderFile.await(),
                        InterpreterApi.Options().apply {
                            runtime = InterpreterApi.Options.TfLiteRuntime.FROM_SYSTEM_ONLY
                        },
                    )

                val decoder =
                    InterpreterApi.create(
                        decoderFile.await(),
                        InterpreterApi.Options().apply {
                            runtime = InterpreterApi.Options.TfLiteRuntime.FROM_SYSTEM_ONLY
                        },
                    )

                TfliteMangaOcr(encoder, decoder)
            }
    }
}
