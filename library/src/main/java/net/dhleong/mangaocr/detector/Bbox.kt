package net.dhleong.mangaocr.detector

import android.graphics.RectF

data class Bbox(
    val rect: RectF,
    val confidence: Float,
)
