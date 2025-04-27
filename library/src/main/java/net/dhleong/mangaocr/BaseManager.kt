package net.dhleong.mangaocr

import android.content.Context
import android.util.Log
import androidx.lifecycle.AtomicReference
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleCoroutineScope
import androidx.lifecycle.repeatOnLifecycle
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import java.io.IOException

abstract class BaseManager<T>(
    context: Context,
    scope: LifecycleCoroutineScope,
    private val lifecycle: Lifecycle,
) {
    sealed interface State<out T> {
        data object Error : State<Nothing>

        data object Loading : State<Nothing>

        data class Loaded<T>(
            val model: T,
        ) : State<T>
    }

    private val context = context.applicationContext
    private val model: AtomicReference<State<T>?> = AtomicReference(null)
    private val modelLoaded = MutableSharedFlow<T>()

    init {
        scope.launch {
            lifecycle.repeatOnLifecycle(Lifecycle.State.RESUMED) {
                try {
                    maybeInit()
                } catch (e: IOException) {
                    return@repeatOnLifecycle
                }
            }
        }
    }

    protected abstract suspend fun initialize(context: Context): T

    protected suspend fun awaitModel(): T = maybeInit()

    private suspend fun maybeInit(): T =
        when (val existing = model.getAndUpdate { it ?: State.Loading }) {
            null, is State.Error -> {
                try {
                    val newModel = initialize(context)
                    model.set(State.Loaded(newModel))
                    modelLoaded.emit(newModel)
                    newModel
                } catch (e: IOException) {
                    // Likely had to download something and couldn't
                    Log.w("BaseManager", "Failed to initialize $this", e)
                    model.set(State.Error)
                    throw e
                }
            }
            is State.Loading -> modelLoaded.first()
            is State.Loaded -> existing.model
        }
}
