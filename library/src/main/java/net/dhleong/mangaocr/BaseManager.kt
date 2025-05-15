package net.dhleong.mangaocr

import android.content.Context
import android.util.Log
import androidx.lifecycle.AtomicReference
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleCoroutineScope
import androidx.lifecycle.repeatOnLifecycle
import kotlinx.coroutines.Dispatchers
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
        // NOTE: The scope may by default use Dispatchers.Main.immediate BUT that means we could
        // initialize the model *before* any properties on subclasses get initialized, so we
        // explicitly request plain Dispatchers.Main to wait at least one frame
        scope.launch(Dispatchers.Main) {
            Log.v("BaseManager", "launching lifecycle scope")
            lifecycle.repeatOnLifecycle(Lifecycle.State.RESUMED) {
                Log.v("BaseManager", "resumed")
                try {
                    maybeInit()
                } catch (e: IOException) {
                    Log.w("BaseManager", "Failed to eagerly initialize manager", e)
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
                    val start = System.currentTimeMillis()
                    val newModel = initialize(context)
                    Log.v("BaseManager", "$this initialized $newModel in ${System.currentTimeMillis() - start}ms (existing=$existing)")

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

    override fun toString(): String = "Manager(${model.get()})"
}
