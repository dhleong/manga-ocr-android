package net.dhleong.mangaocr

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.tooling.preview.Preview
import androidx.lifecycle.lifecycleScope
import coil3.ImageLoader
import coil3.request.ImageRequest
import coil3.request.SuccessResult
import coil3.request.allowHardware
import coil3.toBitmap
import kotlinx.coroutines.launch
import net.dhleong.mangaocr.ui.theme.MangaOCRTheme

private const val USE_REAL_IMAGE = true

class MainActivity : ComponentActivity() {
    private lateinit var manager: MangaOcrManager

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        manager = MangaOcrManager(this, lifecycleScope, lifecycle)

        setContent {
            var lastBitmap: Bitmap? by remember { mutableStateOf(null) }

            MangaOCRTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    Column {
                        Greeting(
                            name = "Android",
                            modifier = Modifier.padding(innerPadding),
                        )

                        if (USE_REAL_IMAGE) {
                            Row {
                                for (i in 0..5) {
                                    Button(onClick = { process(i) { lastBitmap = it } }) {
                                        Text("#$i")
                                    }
                                }
                            }
                        } else {
                            Button(onClick = { process(0) { lastBitmap = it } }) {
                                Text("Hi")
                            }
                        }

                        lastBitmap?.let {
                            val resized = Bitmap.createScaledBitmap(it, 224, 224, true)
                            Image(bitmap = resized.asImageBitmap(), "woah")
                        }
                    }
                }
            }
        }
    }

    private fun process(
        index: Int,
        setBitmap: (Bitmap) -> Unit,
    ) {
        lifecycleScope.launch {
            val bitmap =
                if (!USE_REAL_IMAGE) {
                    Bitmap.createBitmap(256, 256, Bitmap.Config.RGB_565).also {
                        Canvas(it).apply {
                            drawColor(0xffFFFFFF.toInt())
                            drawText(
                                "漫画",
                                128f,
                                128f,
                                Paint().apply {
                                    textAlign = Paint.Align.CENTER
                                    color = 0xff000000.toInt()
                                    textSize = 96f
                                },
                            )
                        }
                    }
                } else {
                    val imageResult =
                        ImageLoader(this@MainActivity).execute(
                            ImageRequest
                                .Builder(this@MainActivity)
                                .data(
                                    "https://github.com/kha-white/manga-ocr/raw/master/assets/examples/0$index.jpg",
                                ).allowHardware(false)
                                .build(),
                        )
                    val image = (imageResult as SuccessResult).image
                    Log.v("OCR", "Loaded $image @ ${image.width} x ${image.height}")
                    image.toBitmap()
                }

            setBitmap(bitmap)
            manager.process(bitmap)
        }
    }
}

@Composable
fun Greeting(
    name: String,
    modifier: Modifier = Modifier,
) {
    Text(
        text = "Hello $name!",
        modifier = modifier,
    )
}

@Preview(showBackground = true)
@Composable
fun GreetingPreview() {
    MangaOCRTheme {
        Greeting("Android")
    }
}
