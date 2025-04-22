package net.dhleong.mangaocr

import android.content.Context
import android.graphics.Bitmap

interface MangaOcr {
    suspend fun process(bitmap: Bitmap)

    companion object {
        suspend fun initialize(context: Context): MangaOcr = OrtMangaOcr.initialize(context)
    }
}
