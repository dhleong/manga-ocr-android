package net.dhleong.mangaocr

import android.content.Context
import android.graphics.Bitmap
import kotlinx.coroutines.flow.Flow
import net.dhleong.mangaocr.ocr.TfliteMangaOcr

interface MangaOcr {
    sealed interface Result {
        data class Partial(
            val text: CharSequence,
        ) : Result

        data class FinalResult(
            val text: String,
        ) : Result
    }

    suspend fun process(bitmap: Bitmap): Flow<Result>

    companion object {
        private const val USE_ORT = false

        suspend fun initialize(context: Context): MangaOcr =
            LoggingMangaOcr(
                if (USE_ORT) {
                    OrtMangaOcr.initialize(context)
                } else {
                    TfliteMangaOcr.initialize(context)
                },
            )
    }
}
