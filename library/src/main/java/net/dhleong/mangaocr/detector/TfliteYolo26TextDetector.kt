package net.dhleong.mangaocr.detector

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import androidx.collection.IntFloatMap
import androidx.collection.intFloatMapOf
import com.google.android.gms.tflite.java.TfLite
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
import kotlin.math.min

/**
 * Old [TfliteMangaTextDetector] detector relied on embedded non-maximum suppression
 * which is no longer supported
 */
class TfliteYolo26TextDetector(
    private val interpreter: InterpreterApi,
    private val model: ModelConfig,
    private val targetWidth: Int = 640,
    private val targetHeight: Int = 640,
    private val processor: Processor = LetterboxProcessor(targetWidth, targetHeight),
) : Detector {
    interface Processor {
        companion object {
            val DEFAULT_TYPE = Type.LETTERBOX
        }

        enum class Type {
            LETTERBOX,
        }

        fun preprocess(bitmap: Bitmap): TensorImage

        fun extractRect(
            output: FloatTensor,
            bitmap: Bitmap,
            index: Int,
        ): RectF
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

            val cxn = output[0, 0, index] * targetWidth
            val cyn = output[0, 1, index] * targetHeight
            val wn = output[0, 2, index] * targetWidth
            val hn = output[0, 3, index] * targetHeight
            val cx = (cxn - padX) / gain
            val cy = (cyn - padY) / gain
            val wHalf = (wn / gain) * 0.5f
            val hHalf = (hn / gain) * 0.5f

            return RectF(cx - wHalf, cy - hHalf, cx + wHalf, cy + hHalf)
        }
    }

    override suspend fun process(bitmap: Bitmap): List<Detector.Result> {
        val processed = processor.preprocess(bitmap)
        val outputTensor = interpreter.getOutputTensor(0)

        Log.v(TAG, "output tensors: ${interpreter.outputTensorCount}")
        Log.v(TAG, "output: ${outputTensor.shape().toList()} ${outputTensor.dataType()}")

        val output = interpreter.allocateFloatOutputTensor(0, rowsCountIndex = 2)
        interpreter.run(processed.buffer, output.buffer)

        Log.v(TAG, "outputRows=${output.rowsCount}")
        return output.mapRows(quitEarlyOnNull = false) { i ->
            val baseClassIndex = 4
            var maxClass = -1
            var maxConfidence = -1f
            this.model.confidenceThresholds.forEach { classIndex, minConfidence ->
                val confidenceIndex = baseClassIndex + classIndex
                val confidence = output[0, confidenceIndex, i]
                Log.v(TAG, "@ $i; class$classIndex has conf $confidence")
                if (confidence >= minConfidence && confidence > maxConfidence) {
                    maxConfidence = confidence
                    maxClass = classIndex
                }
            }
            if (maxClass < 0) {
                return@mapRows null
            }

            val rect = processor.extractRect(output, bitmap, i)
            Detector.Result(
                bbox = Bbox(rect, maxConfidence),
                classIndex = maxClass,
            )
        }
    }

    data class ModelConfig(
        val path: ModelPath,
        val confidenceThresholds: IntFloatMap,
        val processorType: Processor.Type = Processor.DEFAULT_TYPE,
    )

    companion object {
        private const val TAG = "TfliteYolo26TextDetector"

        val YOLO_COCO =
            ModelConfig(
                path =
                    ModelPath(
                        path = "coco-detector-yolos.tflite",
                        sha256 = "066e3a79e587f0ed4de0dfcd7c0d2d8cad856ac7eafdafb92733166f46b0f31c",
                    ),
                confidenceThresholds =
                    intFloatMapOf(
                        0,
                        0.25f, // text
                        1,
                        0.45f, // onomatopoeia
                    ),
            )

        suspend fun initialize(
            context: Context,
            model: ModelConfig = YOLO_COCO,
            // model: ModelConfig = MODEL_INT8_WITH_DATA,
        ): Detector =
            coroutineScope {
                val modelFile =
                    async {
                        HfHubRepo("dhleong/manga-ocr-android")
                            .resolveLocalPath(context, model.path)
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
                    when (model.processorType) {
                        Processor.Type.LETTERBOX -> {
                            LetterboxProcessor(targetWidth, targetHeight)
                        }
                    }
                TfliteYolo26TextDetector(
                    interpreter,
                    model = model,
                    processor = processor,
                )
            }
    }
}
