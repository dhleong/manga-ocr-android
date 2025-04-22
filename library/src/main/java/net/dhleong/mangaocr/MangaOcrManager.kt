package net.dhleong.mangaocr

import android.content.Context
import android.graphics.Bitmap
import androidx.lifecycle.AtomicReference
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleCoroutineScope
import androidx.lifecycle.repeatOnLifecycle
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MangaOcrManager(
    context: Context,
    scope: LifecycleCoroutineScope,
    private val lifecycle: Lifecycle,
) {
    private val model: AtomicReference<MangaOcr?> = AtomicReference(null)
    private val modelLoaded = MutableSharedFlow<Unit>()

    init {
        scope.launch {
            lifecycle.repeatOnLifecycle(Lifecycle.State.RESUMED) {
                if (model.get() == null) {
                    model.set(MangaOcr.initialize(context))
                    modelLoaded.emit(Unit)
                }
            }
        }
    }

    private suspend fun awaitModel(): MangaOcr {
        val existing = model.get()
        if (existing != null) {
            return existing
        }

        modelLoaded.first()
        return requireNotNull(model.get())
    }

    suspend fun process(bitmap: Bitmap) {
        val model = awaitModel()
        withContext(Dispatchers.IO) {
            model.process(bitmap)
        }
    }
}
