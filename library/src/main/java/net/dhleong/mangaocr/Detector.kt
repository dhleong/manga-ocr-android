package net.dhleong.mangaocr

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.android.gms.dynamite.DynamiteModule.LoadingException
import net.dhleong.mangaocr.detector.BBoxHolder
import net.dhleong.mangaocr.detector.Bbox
import net.dhleong.mangaocr.detector.LoggingDetector
import net.dhleong.mangaocr.detector.OrtComicTextDetector
import net.dhleong.mangaocr.detector.TfliteMangaTextDetector
import net.dhleong.mangaocr.detector.TfliteYolo26TextDetector

interface Detector {
    sealed class Type(
        val label: String,
    ) {
        object Legacy : Type(label = "Legacy")

        data class Yolo8(
            val processor: TfliteMangaTextDetector.Processor.Type = TfliteMangaTextDetector.Processor.DEFAULT_TYPE,
        ) : Type(label = "Yolo8 (Old)")

        object YoloCoco : Type(label = "YoloCoco (New)")

        companion object {
            fun iterate() = listOf(Legacy, Yolo8(), YoloCoco)
        }
    }

    suspend fun process(bitmap: Bitmap): List<Result>

    data class Result(
        override val bbox: Bbox,
        val classIndex: Int,
    ) : BBoxHolder

    companion object {
        suspend fun initialize(
            context: Context,
            type: Type,
            fallback: Boolean = true,
        ): Detector =
            LoggingDetector(
                try {
                    when (type) {
                        Type.Legacy -> OrtComicTextDetector.initialize(context)
                        is Type.Yolo8 -> TfliteMangaTextDetector.initialize(context)
                        Type.YoloCoco -> TfliteYolo26TextDetector.initialize(context)
                    }
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
