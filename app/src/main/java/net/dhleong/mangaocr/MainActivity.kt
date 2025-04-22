package net.dhleong.mangaocr

import android.graphics.Bitmap
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
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
            MangaOCRTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    Column {
                        Greeting(
                            name = "Android",
                            modifier = Modifier.padding(innerPadding),
                        )

                        Button(onClick = this@MainActivity::process) {
                            Text("Hi")
                        }
                    }
                }
            }
        }
    }

    private fun process() {
        lifecycleScope.launch {
            manager.process(Bitmap.createBitmap(2048, 2048, Bitmap.Config.RGB_565))
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
