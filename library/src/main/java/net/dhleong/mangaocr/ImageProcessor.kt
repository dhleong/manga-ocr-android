package net.dhleong.mangaocr

// ImageProcessor.kt
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.ColorSpace
import android.os.Build
import android.util.Log
import java.nio.FloatBuffer

class ImageProcessor<T>(
    private val floatsToTensor: (FloatBuffer, shape: LongArray) -> T,
) {
    // Size expected by the ViT model
    private val inputHeight = 224
    private val inputWidth = 224

    // Allocate buffer for tensor
//    private val buffer = FloatBuffer.allocate(3 * inputHeight * inputWidth)
    private val shape = longArrayOf(1, 3, inputHeight.toLong(), inputWidth.toLong())

    fun preprocess(bitmap: Bitmap): T {
        // Convert to RGB if needed

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
                var pixel = resizedBitmap.getPixel(x, y)

//                // NOTE: This *works* but seems less accurate than the original colors?
//                // Translate [0, 1] -> [0, 255]
//                val gray = Color.luminance(pixel) * 255f
                if (GRAYSCALEIFY) {
//                    val gray = (Color.luminance(pixel) * 255f).toInt()
//                    pixel = Color.argb(255, gray, gray, gray)

                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
                        pixel =
                            Color.toArgb(
                                Color.convert(
                                    pixel,
                                    ColorSpace.get(ColorSpace.Named.BT2020_PQ),
                                ),
                            )
                    }
                }
//                val r = gray
//                val g = gray
//                val b = gray

                val r = Color.red(pixel)
                val g = Color.green(pixel)
                val b = Color.blue(pixel)

                val ri = y * inputWidth + x
                val gi = 1 * inputWidth * inputHeight + ri
                val bi = 2 * inputWidth * inputHeight + ri
                buffer.put(ri, (r / 255f - 0.5f) / 0.5f)
                buffer.put(gi, (g / 255f - 0.5f) / 0.5f)
                buffer.put(bi, (b / 255f - 0.5f) / 0.5f)
            }
        }

        if (resizedBitmap !== bitmap) {
            resizedBitmap.recycle()
        }

        // Create tensor with shape [1, 3, 224, 224]
        return floatsToTensor(buffer, shape)
    }

    companion object {
        // Koharu seems to preserve it, but for whatever reason... that doesn't work so well for us
        private const val PRESERVE_ASPECT = false
        private const val GRAYSCALEIFY = true

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
                    Bitmap.createScaledBitmap(this, newWidth, newHeight, true)
                } else {
                    Bitmap.createScaledBitmap(this, inputWidth, inputHeight, true)
                }

            if (resizedBitmap !== this) {
                recycle()
            }
            return resizedBitmap
        }
    }
}
