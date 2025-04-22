package net.dhleong.mangaocr

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Column
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
import kotlinx.coroutines.launch
import net.dhleong.mangaocr.ui.theme.MangaOCRTheme

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

                        Button(onClick = { process { lastBitmap = it } }) {
                            Text("Hi")
                        }

                        lastBitmap?.let {
                            Image(bitmap = it.asImageBitmap(), "woah")
                        }
                    }
                }
            }
        }
    }

    private fun process(setBitmap: (Bitmap) -> Unit) {
        lifecycleScope.launch {
            val bitmap = Bitmap.createBitmap(512, 512, Bitmap.Config.RGB_565)
            Canvas(bitmap).apply {
                drawColor(0xffFFFFFF.toInt())
                drawText(
                    "漫画",
                    256f,
                    256f,
                    Paint().apply {
                        color = 0xff000000.toInt()
                        textSize = 32f
                    },
                )
            }
            manager.process(bitmap)
            setBitmap(bitmap)
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
