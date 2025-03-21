# Steganography Tool - Steganography Analysis Tool

A powerful steganography analysis toolkit developed during my journey through CTF (Capture The Flag) challenges. This tool helped me solve several image steganography puzzles by automating the process of applying various image manipulation techniques.

## Project Background

I developed this tool while working on a particularly challenging steganography puzzle in a CTF competition. The challenge involved detecting hidden information within seemingly innocent image files - a common technique used in CTF competitions and real-world steganography.

The name "Minatori" comes from the word meaning "miners" in Italian, reflecting how this tool helps you mine for hidden data in images.

## Features

- XOR comparison between two images
- Difference blending techniques (add, subtract, darker, lighter)
- Bit plane extraction and analysis
- LSB (Least Significant Bit) extraction
- Channel-specific operations (R, G, B separate analysis)
- Contrast enhancement and thresholding
- Edge detection

## Installation

### Prerequisites
- Python 3.x
- PIL/Pillow
- NumPy

### Setup

1. Clone the repository:
```
git clone https://github.com/Dielldev/Steganography-tool-ctf.git
cd minatori
```

2. Install the required dependencies:
```
pip install pillow numpy
```

## Usage

1. Place your cover image as `cover.gif` in the project directory.
2. Place your suspected steganography image as `results.gif` in the project directory.
3. Run the analysis script:
```
python stego_analysis.py
```
4. Check the `output_images` directory for generated analysis results.

## How It Works

The script applies various image processing techniques to help reveal hidden data:

1. **XOR Operation**: Performs a bitwise XOR between the two images, often revealing hidden content.
2. **Difference Blending**: Shows pixel differences between the images.
3. **Bit Plane Analysis**: Extracts and visualizes each bit plane separately, helpful for finding data hidden in specific bit planes.
4. **Color Channel Analysis**: Isolates differences in specific color channels (R, G, B).
5. **Contrast Enhancement**: Applies extreme contrast to make subtle differences more visible.
6. **LSB Extraction**: Extracts the least significant bit of each pixel, a common place to hide data.

## CTF Tips

When analyzing the output images:
- Pay special attention to bit planes (especially lower bits)
- Look for text or patterns in the XOR and difference results
- Try different thresholds if you see faint patterns
- Check the LSB extraction results carefully
- Sometimes the hidden data is more visible in a specific color channel

## License

MIT

## Credits

Created by [Dielldev] for personal use in CTF competitions. Feel free to use, modify, and extend for your own challenges!

---

*Note: This tool was created for educational purposes and participation in legal CTF competitions. Please use responsibly and ethically.*
