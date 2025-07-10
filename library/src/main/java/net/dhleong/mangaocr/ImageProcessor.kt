package net.dhleong.mangaocr

// ImageProcessor.kt
import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import androidx.core.graphics.get
import androidx.core.graphics.scale
import java.nio.FloatBuffer

class ImageProcessor<T>(
    private val inputHeight: Int = 224,
    private val inputWidth: Int = 224,
    private val normalize: (Float) -> Float = { (it - 0.5f) / 0.5f },
    private val grayscaleify: Boolean = GRAYSCALEIFY,
    private val floatsToTensor: (FloatBuffer, shape: LongArray) -> T,
) {
    // Allocate buffer for tensor
//    private val buffer = FloatBuffer.allocate(3 * inputHeight * inputWidth)
    private val shape = longArrayOf(1, 3, inputHeight.toLong(), inputWidth.toLong())

    fun preprocess(bitmap: Bitmap): T {
        val start = System.currentTimeMillis()

        val resizedBitmap = bitmap.resizeTo(inputWidth, inputHeight)

        Log.v("ImageProcessor", "resized ${bitmap.width} / ${bitmap.height} to ${resizedBitmap.width} / ${resizedBitmap.height}")
        val buffer = FloatBuffer.allocate(3 * inputHeight * inputWidth)
        buffer.clear()
        buffer.limit(buffer.capacity())

        // Convert bitmap to normalized float tensor
        // The shape here seems to be:
        // R[height, width], G[height, width], B[height, width]
        val w = resizedBitmap.width
        val h = resizedBitmap.height
        for (y in 0 until h) {
            for (x in 0 until w) {
                val pixel = resizedBitmap[x, y]
                var r: Int
                var g: Int
                var b: Int

                if (grayscaleify) {
                    // Translate [0, 1] -> [0, 255]
                    val gray = (simpleLuminance(pixel) * 255).toInt()
                    r = gray
                    g = gray
                    b = gray
                } else {
                    r = Color.red(pixel)
                    g = Color.green(pixel)
                    b = Color.blue(pixel)
                }

                val ri = y * inputWidth + x
                val gi = 1 * inputWidth * inputHeight + ri
                val bi = 2 * inputWidth * inputHeight + ri
                buffer.put(ri, normalize(r / 255f))
                buffer.put(gi, normalize(g / 255f))
                buffer.put(bi, normalize(b / 255f))
            }
        }

        if (resizedBitmap !== bitmap) {
            resizedBitmap.recycle()
        }

        val tensor = floatsToTensor(buffer, shape)
        Log.v("ImageProcessor", "preprocessed ($inputWidth x $inputHeight) in ${System.currentTimeMillis() - start}ms")
        return tensor
    }

    companion object {
        // Koharu seems to preserve it, but for whatever reason... that doesn't work so well for us
        private const val PRESERVE_ASPECT = false
        private const val GRAYSCALEIFY = true

        /**
         * Extracts the visual luminance of a pixel Color, similar to [Color.luminance].
         * The primary difference here is that [Color.luminance] does some additional transform
         * on each channel to turn it into the sRGB color space. However, that seems to reduce
         * OCR accuracy for already-black text, so we skip that here and just do the
         * luminance math .
         */
        private fun simpleLuminance(color: Int): Double {
            // NOTE: Color.luminance also converts the color [0, 255] -> [0, 1] *before*
            // doing the luminance extraction below. However, anecdotally, doing it *after*
            // that extraction seems to preserve slightly more accuracy.
            val r = Color.red(color)
            val g = Color.green(color)
            val b = Color.blue(color)

            val lr = (0.2126 * r) / 255.0
            val lg = (0.7152 * g) / 255.0
            val lb = (0.0722 * b) / 255.0

            return lr + lg + lb
        }

        fun Bitmap.resizeTo(
            inputWidth: Int,
            inputHeight: Int,
        ): Bitmap {
            val resizedBitmap =
                if (PRESERVE_ASPECT) {
                    val scale: Float =
                        if (width > height) {
                            inputWidth.toFloat() / width
                        } else {
                            inputHeight.toFloat() / height
                        }

                    val newWidth = (width * scale).toInt()
                    val newHeight = (height * scale).toInt()
                    scale(newWidth, newHeight)
                } else {
                    scale(inputWidth, inputHeight)
                }

            return resizedBitmap
        }
    }
}
