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
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import java.nio.ByteBuffer
import java.nio.ByteOrder
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

        fun preprocess(bitmap: Bitmap): ByteBuffer

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
                .build()

        override fun preprocess(bitmap: Bitmap): ByteBuffer {
            val tensorImage = imageProcessor.process(TensorImage.fromBitmap(bitmap))
            val letterboxed = tensorImage.bitmap

            val pixels = IntArray(targetWidth * targetHeight)
            letterboxed.getPixels(pixels, 0, targetWidth, 0, 0, targetWidth, targetHeight)

            val byteBuffer =
                ByteBuffer
                    .allocateDirect(1 * 3 * targetHeight * targetWidth * 4)
                    .order(ByteOrder.nativeOrder())
            val floatBuffer = byteBuffer.asFloatBuffer()

            val area = targetWidth * targetHeight
            val norm = 1f / 255f
            for (i in pixels.indices) {
                val pixel = pixels[i]
                floatBuffer.put(i, ((pixel shr 16) and 0xFF) * norm)
                floatBuffer.put(area + i, ((pixel shr 8) and 0xFF) * norm)
                floatBuffer.put(2 * area + i, (pixel and 0xFF) * norm)
            }
            byteBuffer.rewind()
            return byteBuffer
        }

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
        interpreter.run(processed, output.buffer)

        Log.v(TAG, "outputRows=${output.rowsCount}")
        val baseClassIndex = 4

        return output
            .sequenceFromRows(
                quitEarlyOnNull = false,
            ) { i ->
                var maxClass = -1
                var maxConfidence = -1f
                this.model.confidenceThresholds.forEach { classIndex, minConfidence ->
                    val confidenceIndex = baseClassIndex + classIndex
                    val confidence = output[0, confidenceIndex, i]
                    if (confidence >= minConfidence && confidence > maxConfidence) {
                        maxConfidence = confidence
                        maxClass = classIndex
                    }
                }
                if (maxClass >= 0) {
                    val rect = processor.extractRect(output, bitmap, i)
                    Detector.Result(classIndex = maxClass, bbox = Bbox(rect, maxConfidence))
                } else {
                    null
                }
            }.groupBy { it.classIndex }
            .flatMap { (_, boxes) ->
                nonMaximumSuppression(listOf(boxes), threshold = model.nmsThreshold).first()
            }
    }

    data class ModelConfig(
        val path: ModelPath,
        val confidenceThresholds: IntFloatMap,
        val nmsThreshold: Float = DEFAULT_NMS_THRESHOLD,
        val processorType: Processor.Type = Processor.DEFAULT_TYPE,
    )

    companion object {
        private const val TAG = "TfliteYolo26TextDetector"
        private const val DEFAULT_NMS_THRESHOLD = 0.5f

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
                        0.15f, // text
                        1,
                        0.45f, // onomatopoeia
                    ),
            )

        val YOLO_COCO_QUANTIZED =
            YOLO_COCO.copy(
                path =
                    ModelPath(
                        path = "coco-detector-yolos-w8a32.tflite",
                        sha256 = "38ee43a8bafc7ab1148b0c9ab5a52b4f4ccc0727245240ddb23f2f9943123aa0",
                    ),
            )

        suspend fun initialize(
            context: Context,
            model: ModelConfig = YOLO_COCO_QUANTIZED,
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
