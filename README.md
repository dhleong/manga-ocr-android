# manga-ocr-android

*Manga OCR made fast*

## What?

This repository is an attempt to port the work from [manga-ocr][manga-ocr] and [koharu][koharu] to Android, for integration into [Mihon][mihon]. See the [model-dev README](/model-dev/README.md) for more details on the model work.

## How?

The library module is published via jitpack:

```
dependencies {
    implementation 'com.github.dhleong:manga-ocr-android:a4f2a6c8f0'
}
```

The required models are downloaded automatically when first used.

## Is it really fast?

I haven't done any real benchmarking, but anecdotally—yes! A quantized version of the comictextdetector model from [koharu][koharu] took about 3s on average to detect text boxes; the default `Detector` implementation in the library works in about 300 *milliseconds*.

A quantized version of the [manga-ocr][manga-ocr] model took about 110ms *per character* in the box. The current `MangaOcr` implementation in the library only takes about 10ms per character.

[manga-ocr]: https://github.com/kha-white/manga-ocr
[koharu]: https://github.com/mayocream/koharu
[mihon]: https://mihon.app
