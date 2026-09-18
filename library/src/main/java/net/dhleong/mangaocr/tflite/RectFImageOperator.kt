package net.dhleong.mangaocr.tflite

import android.graphics.RectF

interface RectFImageOperator {
    fun inverseTransform(
        rect: RectF,
        inputImageHeight: Int,
        inputImageWidth: Int,
    ): RectF
}
