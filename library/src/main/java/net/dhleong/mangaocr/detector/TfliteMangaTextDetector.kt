package net.dhleong.mangaocr.detector

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import com.google.android.gms.tflite.java.TfLite
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope
import net.dhleong.mangaocr.Detector
import net.dhleong.mangaocr.hub.HfHubRepo
import net.dhleong.mangaocr.hub.ModelPath
import net.dhleong.mangaocr.onnx.FloatTensor.Companion.allocateFloatOutputTensor
import net.dhleong.mangaocr.tflite.await
import org.tensorflow.lite.InterpreterApi
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp

class TfliteMangaTextDetector(
    private val interpreter: InterpreterApi,
    private val targetWidth: Int = 640,
    private val targetHeight: Int = 640,
) : Detector {
    override suspend fun process(bitmap: Bitmap): List<Detector.Result> {
        val imageProcessor =
            ImageProcessor
                .Builder()
                .add(ResizeOp(targetHeight, targetWidth, ResizeOp.ResizeMethod.BILINEAR))
                .add(NormalizeOp(0f, 255f)) // [ 0, 1 ]
                .build()
        val processed = imageProcessor.process(TensorImage.fromBitmap(bitmap))

        // The YOLO model seems to output in xyxyn format,
        // IE: normalized within the *original* width
        val ratioX = bitmap.width
        val ratioY = bitmap.height

        val outputTensor = interpreter.getOutputTensor(0)
        Log.v("TfliteDetector", "output: ${outputTensor.shape().toList()} ${outputTensor.dataType()}")

        val output = interpreter.allocateFloatOutputTensor(0)
        interpreter.run(processed.buffer, output.buffer)
        Log.v("TfliteDetector", "output: ${output.buffer.array().toList()}")

        return output.mapRows { i ->
            val confidence = output[0, i, 4]
            if (confidence < CONFIDENCE_THRESHOLD) {
                return@mapRows null
            }

            val left = output[0, i, 0] * ratioX
            val top = output[0, i, 1] * ratioY
            val right = output[0, i, 2] * ratioX
            val bottom = output[0, i, 3] * ratioY

            Detector.Result(
                bbox = Bbox(RectF(left, top, right, bottom), confidence),
                classIndex = output[0, i, 5].toInt(),
            )
        }
    }

    companion object {
        private const val CONFIDENCE_THRESHOLD = 0.05f

        @Suppress("unused")
        val MODEL_FLOAT32 =
            ModelPath(
                path = "manga-text-detector_float32.tflite",
                sha256 = "cc8f6894424cebaeda7b3950004b77838967ba28e6d3b01bf94025689eebde40",
            )

        @Suppress("unused")
        val MODEL_FLOAT16 =
            ModelPath(
                path = "manga-text-detector_float16.tflite",
                sha256 = "ede78ec00d546528e85a743c65b4c7f1614b7271526149434674656f6e7c8ce1",
            )

        private val MODEL_INT8 =
            ModelPath(
                path = "manga-text-detector_int8.tflite",
                sha256 = "2c8a423844cfb4707f26a1e0d493a8919315a2dfea079c2f1fdb5cf8e55a1f60",
            )

        suspend fun initialize(
            context: Context,
            model: ModelPath = MODEL_INT8,
        ): Detector =
            coroutineScope {
                val modelFile =
                    async {
                        HfHubRepo("dhleong/manga-ocr-android")
                            .resolveLocalPath(context, model)
                    }

                val initialized = async { TfLite.initialize(context).await() }

                initialized.await()
                val interpreter =
                    InterpreterApi.create(
                        modelFile.await(),
                        InterpreterApi.Options().apply {
                            runtime = InterpreterApi.Options.TfLiteRuntime.FROM_SYSTEM_ONLY
                        },
                    )

                TfliteMangaTextDetector(interpreter)
            }
    }
}
