package net.dhleong.mangaocr

import android.graphics.Bitmap
import android.graphics.RectF
import net.dhleong.mangaocr.tflite.RectFImageOperator
import net.dhleong.mangaocr.tflite.TransposeToNchwOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageOperator
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import java.nio.FloatBuffer

class ImageProcessor2<T>(
    private val outputFormat: OutputFormat,
    private val inputHeight: Int,
    private val inputWidth: Int,
    private val resizeOp: ImageOperator,
    private val normalizeOp: NormalizeOp? = null,
    private val bboxFormat: BboxFormat? = null,
    private val floatsToTensor: (FloatBuffer, shape: LongArray) -> T,
) {
    private val shape = longArrayOf(1, inputHeight.toLong(), inputWidth.toLong(), 3)

    private val processor: ImageProcessor =
        ImageProcessor
            .Builder()
            .apply {
                add(resizeOp)

                when (outputFormat) {
                    OutputFormat.NHWC -> { /* nop */ }

                    OutputFormat.NCHW -> {
                        add(TransposeToNchwOp())
                    }
                }

                normalizeOp?.let(::add)
            }.build()

    fun process(bitmap: Bitmap): T {
        val processed = processor.process(TensorImage.fromBitmap(bitmap))

        return floatsToTensor(processed.buffer.asFloatBuffer(), shape)
    }

    fun inverseTransform(
        bitmap: Bitmap,
        a: Float,
        b: Float,
        c: Float,
        d: Float,
    ): RectF {
        val format =
            this.bboxFormat
                ?: throw UnsupportedOperationException("No bboxFormat provided")
        val normalized = format.extractRectF(a, b, c, d)
        val op = resizeOp
        return if (op is RectFImageOperator) {
            op.inverseTransform(normalized, bitmap.height, bitmap.width)
        } else {
            this.processor.inverseTransform(normalized, bitmap.height, bitmap.width)
        }
    }

    enum class OutputFormat {
        // [N, height, width, channels]
        // This is the preferred format of tflite, and
        // supported by TensorImage directly
        NHWC,

        // [N, channels, height, width]
        NCHW,
    }

    sealed interface BboxFormat {
        fun extractRectF(
            a: Float,
            b: Float,
            c: Float,
            d: Float,
        ): RectF

        /**
         * `[a, b]` represent cx, cy—the coordinates of the center of the box
         * `[c, d]` represent w, h—the size of the box
         */
        object CenterAndSize : BboxFormat {
            override fun extractRectF(
                a: Float,
                b: Float,
                c: Float,
                d: Float,
            ): RectF {
                val wHalf = c * 0.5f
                val hHalf = d * 0.5f
                return RectF(a - wHalf, b - hHalf, a + wHalf, b + hHalf)
            }
        }
    }
}
