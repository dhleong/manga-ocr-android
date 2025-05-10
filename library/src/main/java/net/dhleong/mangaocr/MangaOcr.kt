package net.dhleong.mangaocr

import android.content.Context
import android.graphics.Bitmap
import kotlinx.coroutines.flow.Flow
import net.dhleong.mangaocr.ocr.SplitPhaseMangaOcr

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
        /**
         * @param useCombinedModel There's probably no reason to use this. It's an order of
         * magnitude slower than the split-phase model. But it's made available just in case.
         */
        suspend fun initialize(
            context: Context,
            useCombinedModel: Boolean = false,
        ): MangaOcr =
            LoggingMangaOcr(
                if (useCombinedModel) {
                    OrtMangaOcr.initialize(context)
                } else {
                    SplitPhaseMangaOcr.initialize(context)
                },
            )
    }
}
