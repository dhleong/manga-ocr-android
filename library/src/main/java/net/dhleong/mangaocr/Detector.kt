package net.dhleong.mangaocr

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.android.gms.dynamite.DynamiteModule.LoadingException
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
        suspend fun initializeLegacy(context: Context): Detector = LoggingDetector(OrtComicTextDetector.initialize(context))

        suspend fun initialize(
            context: Context,
            processorType: TfliteMangaTextDetector.Processor.Type = TfliteMangaTextDetector.Processor.DEFAULT_TYPE,
            fallback: Boolean = true,
        ): Detector =
            LoggingDetector(
                try {
                    TfliteMangaTextDetector.initialize(context, processorType = processorType)
                } catch (e: LoadingException) {
                    if (!fallback) {
                        throw e
                    }

                    Log.w(
                        "manga-ocr-android",
                        "Failed to initialize tflite detector; falling back",
                        e,
                    )
                    OrtComicTextDetector.initialize(context)
                },
            )
    }
}
