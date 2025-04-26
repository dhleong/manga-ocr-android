package net.dhleong.mangaocr

import android.content.Context
import android.graphics.Bitmap
import net.dhleong.mangaocr.detector.Bbox
import net.dhleong.mangaocr.detector.LoggingDetector
import net.dhleong.mangaocr.detector.OrtComicTextDetector

interface Detector {
    suspend fun process(bitmap: Bitmap): List<Result>

    data class Result(
        val bbox: Bbox,
        val classIndex: Int,
    )

    companion object {
        suspend fun initialize(context: Context): Detector = LoggingDetector(OrtComicTextDetector.initialize(context))
    }
}
