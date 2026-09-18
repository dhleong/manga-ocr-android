package net.dhleong.mangaocr.tflite

import android.annotation.SuppressLint
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.PointF
import android.graphics.Rect
import android.graphics.RectF
import org.tensorflow.lite.support.common.internal.SupportPreconditions
import org.tensorflow.lite.support.image.ColorSpaceType
import org.tensorflow.lite.support.image.ImageOperator
import org.tensorflow.lite.support.image.TensorImage
import kotlin.math.min

class ResizeWithPadOp(
    private val targetHeight: Int,
    private val targetWidth: Int,
) : ImageOperator,
    RectFImageOperator {
    @SuppressLint("UseKtx")
    private val output =
        Bitmap.createBitmap(
            this.targetWidth,
            this.targetHeight,
            Bitmap.Config.ARGB_8888,
        )

    override fun apply(image: TensorImage): TensorImage {
        SupportPreconditions.checkArgument(
            image.colorSpaceType === ColorSpaceType.RGB,
            "Only RGB images are supported in ResizeWithCropOrPadOp, but not " + image.colorSpaceType.name,
        )
        val input = image.bitmap
        val w = input.width
        val h = input.height

        val scaleFactorW = this.targetWidth.toFloat() / w
        val scaleFactorH = this.targetHeight.toFloat() / h
        val scaleFactor = minOf(scaleFactorW, scaleFactorH)

        val scaledW = (w * scaleFactor).toInt()
        val scaledH = (h * scaleFactor).toInt()

        val dstL: Int
        val dstR: Int
        if (this.targetWidth > scaledW) {
            dstL = (this.targetWidth - scaledW) / 2
            dstR = dstL + scaledW
        } else {
            dstL = 0
            dstR = this.targetWidth
        }

        val dstT: Int
        val dstB: Int
        if (this.targetHeight > scaledH) {
            dstT = (this.targetHeight - scaledH) / 2
            dstB = dstT + scaledH
        } else {
            dstT = 0
            dstB = this.targetHeight
        }

        val src = Rect(0, 0, w, h)
        val dst = Rect(dstL, dstT, dstR, dstB)

        Canvas(output).apply {
            drawColor(0xff000000.toInt())
            drawBitmap(input, src, dst, null as Paint?)
        }
        image.load(this.output)
        return image
    }

    override fun getOutputImageHeight(
        inputImageHeight: Int,
        inputImageWidth: Int,
    ): Int = this.targetHeight

    override fun getOutputImageWidth(
        inputImageHeight: Int,
        inputImageWidth: Int,
    ): Int = this.targetWidth

    override fun inverseTransform(
        point: PointF,
        inputImageHeight: Int,
        inputImageWidth: Int,
    ): PointF {
        val gain =
            min(
                targetHeight / inputImageHeight.toFloat(),
                targetWidth / inputImageWidth.toFloat(),
            )
        val padX =
            (targetWidth - inputImageWidth * gain) / 2 - 0.1f
        val padY =
            (targetHeight - inputImageHeight * gain) / 2 - 0.1f

        val x = (point.x - padX) / gain
        val y = (point.y - padY) / gain
        return PointF(x, y)
    }

    override fun inverseTransform(
        rect: RectF,
        inputImageHeight: Int,
        inputImageWidth: Int,
    ): RectF {
        val gain =
            min(
                targetHeight / inputImageHeight.toFloat(),
                targetWidth / inputImageWidth.toFloat(),
            )
        val padX =
            (targetWidth - inputImageWidth * gain) / 2 - 0.1f
        val padY =
            (targetHeight - inputImageHeight * gain) / 2 - 0.1f

        val cx = (rect.centerX() * targetWidth - padX) / gain
        val cy = (rect.centerY() * targetHeight - padY) / gain
        val wHalf = (rect.width() * targetWidth * 0.5f) / gain
        val hHalf = (rect.height() * targetHeight * 0.5f) / gain
        return RectF(cx - wHalf, cy - hHalf, cx + wHalf, cy + hHalf)
    }
}
