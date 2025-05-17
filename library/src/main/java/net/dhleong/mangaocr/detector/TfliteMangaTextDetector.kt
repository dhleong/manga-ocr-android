package net.dhleong.mangaocr.detector

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import com.google.android.gms.tflite.java.TfLite
import com.google.common.primitives.Floats.min
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope
import net.dhleong.mangaocr.Detector
import net.dhleong.mangaocr.hub.HfHubRepo
import net.dhleong.mangaocr.hub.ModelPath
import net.dhleong.mangaocr.onnx.FloatTensor
import net.dhleong.mangaocr.onnx.FloatTensor.Companion.allocateFloatOutputTensor
import net.dhleong.mangaocr.tflite.ResizeWithPadOp
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
    private val processor: Processor = LetterboxProcessor(targetWidth, targetHeight),
) : Detector {
    interface Processor {
        enum class Type {
            OLD,
            LETTERBOX,
        }

        fun preprocess(bitmap: Bitmap): TensorImage

        fun extractRect(
            output: FloatTensor,
            bitmap: Bitmap,
            index: Int,
        ): RectF
    }

    class OldProcessor(
        targetWidth: Int,
        targetHeight: Int,
    ) : Processor {
        private val imageProcessor =
            ImageProcessor
                .Builder()
                .add(ResizeOp(targetHeight, targetWidth, ResizeOp.ResizeMethod.BILINEAR))
                .add(NormalizeOp(0f, 255f)) // [ 0, 1 ]
                .build()

        override fun preprocess(bitmap: Bitmap): TensorImage = imageProcessor.process(TensorImage.fromBitmap(bitmap))

        override fun extractRect(
            output: FloatTensor,
            bitmap: Bitmap,
            index: Int,
        ): RectF {
            // The YOLO model seems to output in xyxyn format,
            // IE: normalized within the *original* width
            val left = output[0, index, 0] * bitmap.width
            val top = output[0, index, 1] * bitmap.height
            val right = output[0, index, 2] * bitmap.width
            val bottom = output[0, index, 3] * bitmap.height
            return RectF(left, top, right, bottom)
        }
    }

    class LetterboxProcessor(
        private val targetWidth: Int,
        private val targetHeight: Int,
    ) : Processor {
        private val resize = ResizeWithPadOp(targetHeight, targetWidth)
        private val imageProcessor =
            ImageProcessor
                .Builder()
                .add(resize)
                .add(NormalizeOp(0f, 255f)) // [ 0, 1 ]
                .build()

        override fun preprocess(bitmap: Bitmap): TensorImage = imageProcessor.process(TensorImage.fromBitmap(bitmap))

        override fun extractRect(
            output: FloatTensor,
            bitmap: Bitmap,
            index: Int,
        ): RectF {
            val gain =
                min(
                    targetHeight / bitmap.height.toFloat(),
                    targetWidth / bitmap.width.toFloat(),
                )
            val padX =
                (targetWidth - bitmap.width * gain) / 2 - 0.1f
            val padY =
                (targetHeight - bitmap.height * gain) / 2 - 0.1f

            val xn = output[0, index, 0] * targetWidth
            val yn = output[0, index, 1] * targetHeight
            val xm = output[0, index, 2] * targetWidth
            val ym = output[0, index, 3] * targetHeight
            val left = (xn - padX) / gain
            val top = (yn - padY) / gain
            val right = (xm - padX) / gain
            val bottom = (ym - padY) / gain
            Log.v(
                "TfliteDetector",
                "gain: $gain pad: $padX, $padY (${bitmap.width} x ${bitmap.height})" +
                    " -> ${output[0, index, 0]} $xn $yn $xm $ym" +
                    " => $left $top $right $bottom",
            )

            return RectF(left, top, right, bottom)
        }
    }

    override suspend fun process(bitmap: Bitmap): List<Detector.Result> {
        val processed = processor.preprocess(bitmap)
        val outputTensor = interpreter.getOutputTensor(0)
        Log.v("TfliteDetector", "output: ${outputTensor.shape().toList()} ${outputTensor.dataType()}")

        val output = interpreter.allocateFloatOutputTensor(0)
        interpreter.run(processed.buffer, output.buffer)

        return output.mapRows { i ->
            val confidence = output[0, i, 4]
            if (confidence < CONFIDENCE_THRESHOLD) {
                return@mapRows null
            }

            val rect = processor.extractRect(output, bitmap, i)
            Detector.Result(
                bbox = Bbox(rect, confidence),
                classIndex = output[0, i, 5].toInt(),
            )
        }
    }

    companion object {
        private const val CONFIDENCE_THRESHOLD = 0.25f

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

        private val MODEL_INT8_WITH_DATA =
            ModelPath(
                path = "manga-text-detector_int8.with_data.tflite",
                sha256 = "",
            )

        suspend fun initialize(
            context: Context,
            model: ModelPath = MODEL_INT8_WITH_DATA,
            processorType: Processor.Type = Processor.Type.LETTERBOX,
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

                val targetWidth = 640
                val targetHeight = 640
                val processor =
                    when (processorType) {
                        Processor.Type.OLD -> OldProcessor(targetWidth, targetHeight)
                        Processor.Type.LETTERBOX -> LetterboxProcessor(targetWidth, targetHeight)
                    }
                TfliteMangaTextDetector(interpreter, processor = processor)
            }
    }
}
