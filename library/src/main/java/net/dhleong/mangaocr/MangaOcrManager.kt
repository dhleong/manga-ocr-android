package net.dhleong.mangaocr

import android.content.Context
import android.graphics.Bitmap
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleCoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.withContext

class MangaOcrManager(
    context: Context,
    scope: LifecycleCoroutineScope,
    lifecycle: Lifecycle,
) : BaseManager<MangaOcr>(context, scope, lifecycle),
    MangaOcr {
    override suspend fun initialize(context: Context): MangaOcr = MangaOcr.initialize(context)

    override suspend fun process(bitmap: Bitmap): Flow<MangaOcr.Result> {
        val model = awaitModel()
        return withContext(Dispatchers.IO) {
            model.process(bitmap)
        }
    }
}
