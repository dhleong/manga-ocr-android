package net.dhleong.mangaocr

import android.content.Context
import android.graphics.Bitmap
import net.dhleong.mangaocr.detector.Bbox
import net.dhleong.mangaocr.detector.LoggingDetector
import net.dhleong.mangaocr.detector.OrtComicTextDetector
import net.dhleong.mangaocr.detector.TfliteMangaTextDetector

interface Detector {
    suspend fun process(bitmap: Bitmap): List<Result>

    data class Result(
        val bbox: Bbox,
        val classIndex: Int,
    )

    companion object {
        private const val USE_TFLITE = true

        suspend fun initialize(context: Context): Detector =
            LoggingDetector(
                if (USE_TFLITE) {
                    TfliteMangaTextDetector.initialize(context)
                } else {
                    OrtComicTextDetector.initialize(context)
                },
            )
    }
}
