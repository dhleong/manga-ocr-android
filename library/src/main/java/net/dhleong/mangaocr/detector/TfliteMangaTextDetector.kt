package net.dhleong.mangaocr.detector

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import androidx.collection.IntFloatMap
import androidx.collection.IntSet
import androidx.collection.intFloatMapOf
import androidx.collection.intSetOf
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
            OLD,
            LETTERBOX,
            LETTERBOX_SELECTIVE,
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

            return RectF(left, top, right, bottom)
        }
    }

    override suspend fun process(bitmap: Bitmap): List<Detector.Result> {
        val processed = processor.preprocess(bitmap)
        val outputTensor = interpreter.getOutputTensor(0)

        Log.v("TfliteDetector", "output tensors: ${interpreter.outputTensorCount}")
        Log.v("TfliteDetector", "output: ${outputTensor.shape().toList()} ${outputTensor.dataType()}")

        val output = interpreter.allocateFloatOutputTensor(0)
        interpreter.run(processed.buffer, output.buffer)

        val rows =
            output.mapRows(quitEarlyOnNull = false) { i ->
                val confidence = output[0, i, 4]
                val classIndex = output[0, i, 5].toInt()
                val threshold =
                    model.confidenceThresholds?.getOrDefault(classIndex, model.defaultConfidenceThreshold)
                        ?: model.defaultConfidenceThreshold
                if (confidence < threshold) {
                    return@mapRows null
                }

                val rect = processor.extractRect(output, bitmap, i)
                Detector.Result(
                    bbox = Bbox(rect, confidence),
                    classIndex = classIndex,
                )
            }
        return model.allowedClassIndices?.let { indices ->
            rows.filter { it.classIndex in indices }
        } ?: rows
    }

    data class ModelConfig(
        val path: ModelPath,
        val defaultConfidenceThreshold: Float,
        val confidenceThresholds: IntFloatMap? = null,
        val processorType: Processor.Type = Processor.DEFAULT_TYPE,
        val allowedClassIndices: IntSet? = null,
    )

    companion object {
        private val MODEL_INT8_WITH_DATA =
            ModelConfig(
                path =
                    ModelPath(
                        path = "manga-text-detector_int8.with_data.tflite",
                        sha256 = "2bc1213c7dc666d326f1b6c5a74adc62a1e946cec8b607d146164c7e85dcaf71",
                    ),
                defaultConfidenceThreshold = 0.25f,
                confidenceThresholds = intFloatMapOf(0, 0.7f),
            )

        @Suppress("unused")
        val MODEL_INT8_WITH_DATA_SELECTIVE =
            MODEL_INT8_WITH_DATA.copy(
                processorType = Processor.Type.LETTERBOX_SELECTIVE,
                // For this model, 1 is the "primary class"
                allowedClassIndices = intSetOf(1),
            )

        val YOLO_COCO =
            ModelConfig(
                path =
                    ModelPath(
                        path = "coco-detector-yolos.tflite",
                        sha256 = "066e3a79e587f0ed4de0dfcd7c0d2d8cad856ac7eafdafb92733166f46b0f31c",
                    ),
                defaultConfidenceThreshold = 0.25f,
                confidenceThresholds = intFloatMapOf(1, 0.45f),
                allowedClassIndices =
                    intSetOf(
                        0, // text
                        1, // onomatopoeia
                        // NOTE: 3 is "bubble" which is so far duplicative
                    ),
            )

        suspend fun initialize(
            context: Context,
            // model: ModelConfig = YOLO_COCO,
           model: ModelConfig = MODEL_INT8_WITH_DATA,
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
                        Processor.Type.OLD -> {
                            OldProcessor(targetWidth, targetHeight)
                        }

                        Processor.Type.LETTERBOX, Processor.Type.LETTERBOX_SELECTIVE -> {
                            LetterboxProcessor(targetWidth, targetHeight)
                        }
                    }
                TfliteMangaTextDetector(
                    interpreter,
                    model = model,
                    processor = processor,
                )
            }
    }
}
