from PIL import Image, ImageEnhance, ImageFilter, ImageChops, ImageOps
import numpy as np
import os


output_dir = "output_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the images
image1_path = "cover.gif"
image2_path = "results.gif"

image1 = Image.open(image1_path)
image2 = Image.open(image2_path)

print(f"Image 1 mode: {image1.mode}, size: {image1.size}")
print(f"Image 2 mode: {image2.mode}, size: {image2.size}")


if image1.size != image2.size:
    image2 = image2.resize(image1.size)


if image1.mode != 'RGB':
    image1 = image1.convert('RGB')
if image2.mode != 'RGB':
    image2 = image2.convert('RGB')


array1 = np.array(image1)
array2 = np.array(image2)


xor_result = np.bitwise_xor(array1, array2)
xor_image = Image.fromarray(xor_result)
xor_image.save(f"{output_dir}/1_xor_result.png")


diff_image = ImageChops.difference(image1, image2)
diff_image.save(f"{output_dir}/2_difference_blend.png")


add_image = ImageChops.add(image1, image2, scale=2.0, offset=0)
add_image.save(f"{output_dir}/3_add_blend.png")


darker_image = ImageChops.darker(image1, image2)
darker_image.save(f"{output_dir}/4_darker_blend.png")

lighter_image = ImageChops.lighter(image1, image2)
lighter_image.save(f"{output_dir}/5_lighter_blend.png")


def extract_bit_planes(img_array):
    bit_planes = []
    for bit in range(8):
        # Extract the bit plane
        plane = np.bitwise_and(img_array, 2**bit) // 2**bit
        plane = plane * 255  # Scale to full intensity for visibility
        bit_planes.append(plane.astype(np.uint8))
    return bit_planes

# Process XOR result for bit planes
xor_bit_planes = extract_bit_planes(xor_result)
for i, plane in enumerate(xor_bit_planes):
    bit_image = Image.fromarray(plane)
    bit_image.save(f"{output_dir}/6_xor_bit_plane_{i}.png")

# 4. Try different bit-level operations
# AND operation
and_result = np.bitwise_and(array1, array2)
and_image = Image.fromarray(and_result)
and_image.save(f"{output_dir}/7_and_result.png")

# OR operation
or_result = np.bitwise_or(array1, array2)
or_image = Image.fromarray(or_result)
or_image.save(f"{output_dir}/8_or_result.png")

# 5. Try subtracting specific color channels
def subtract_channels(img1, img2, subtract_mode='rgb'):
    result = np.zeros_like(img1)
    if subtract_mode == 'r':
        # Subtract only red channel
        result[:,:,0] = np.abs(img1[:,:,0] - img2[:,:,0])
        result[:,:,1] = img1[:,:,1]
        result[:,:,2] = img1[:,:,2]
    elif subtract_mode == 'g':
        # Subtract only green channel
        result[:,:,0] = img1[:,:,0]
        result[:,:,1] = np.abs(img1[:,:,1] - img2[:,:,1])
        result[:,:,2] = img1[:,:,2]
    elif subtract_mode == 'b':
        # Subtract only blue channel
        result[:,:,0] = img1[:,:,0]
        result[:,:,1] = img1[:,:,1]
        result[:,:,2] = np.abs(img1[:,:,2] - img2[:,:,2])
    else:
        # Subtract all channels
        result = np.abs(img1 - img2)
    return result

# Subtract each channel individually
r_subtract = subtract_channels(array1, array2, 'r')
g_subtract = subtract_channels(array1, array2, 'g')
b_subtract = subtract_channels(array1, array2, 'b')
all_subtract = subtract_channels(array1, array2, 'rgb')

Image.fromarray(r_subtract).save(f"{output_dir}/9_red_subtract.png")
Image.fromarray(g_subtract).save(f"{output_dir}/10_green_subtract.png")
Image.fromarray(b_subtract).save(f"{output_dir}/11_blue_subtract.png")
Image.fromarray(all_subtract).save(f"{output_dir}/12_all_subtract.png")

# 6. Apply contrast enhancement to the most promising results
for i, img_array in enumerate([xor_result, all_subtract]):
    img = Image.fromarray(img_array)
    
    # High contrast
    contrast_enhancer = ImageEnhance.Contrast(img)
    high_contrast = contrast_enhancer.enhance(10.0)
    high_contrast.save(f"{output_dir}/13_high_contrast_{i}.png")
    
    # Apply threshold after contrast
    gray_hc = high_contrast.convert('L')
    # Try multiple threshold values
    for threshold in [100, 128, 150, 175]:
        threshold_img = gray_hc.point(lambda x: 0 if x < threshold else 255, '1')
        threshold_img.save(f"{output_dir}/14_threshold_{i}_{threshold}.png")
    
    # Invert and try again - sometimes text is in reverse contrast
    inverted = ImageOps.invert(img)
    inverted.save(f"{output_dir}/15_inverted_{i}.png")
    inv_contrast = ImageEnhance.Contrast(inverted).enhance(10.0)
    inv_contrast.save(f"{output_dir}/16_inverted_contrast_{i}.png")

# 7. Apply edge detection using PIL instead of OpenCV
for i, img_array in enumerate([xor_result, all_subtract]):
    img = Image.fromarray(img_array).convert('L')  # Convert to grayscale
    
    # Use PIL's built-in FIND_EDGES filter
    edge_img = img.filter(ImageFilter.FIND_EDGES)
    edge_img.save(f"{output_dir}/17_edge_detection_{i}.png")
    
    # Enhance edges further
    edge_enhanced = ImageEnhance.Contrast(edge_img).enhance(2.0)
    edge_enhanced.save(f"{output_dir}/18_edge_enhanced_{i}.png")

# 8. Try combining operations for better results
# XOR result with high pass filter (sharpening)
xor_sharp = Image.fromarray(xor_result).filter(ImageFilter.SHARPEN)
xor_sharp = xor_sharp.filter(ImageFilter.SHARPEN)  # Apply twice for stronger effect
xor_sharp.save(f"{output_dir}/19_xor_sharpened.png")

# Try extreme brightness/contrast manipulations
for i, img_array in enumerate([xor_result, all_subtract]):
    img = Image.fromarray(img_array)
    
    # Multiple enhancement steps in sequence
    enhanced = img
    enhanced = ImageEnhance.Contrast(enhanced).enhance(15.0)  # Very high contrast
    enhanced = ImageEnhance.Brightness(enhanced).enhance(1.1)  # Slight brightness boost
    enhanced = enhanced.filter(ImageFilter.SHARPEN)
    enhanced.save(f"{output_dir}/20_extreme_enhance_{i}.png")
    
    # Convert to grayscale and apply threshold with various levels
    gray = enhanced.convert('L')
    for threshold in [100, 125, 150, 175, 200]:
        threshold_img = gray.point(lambda x: 0 if x < threshold else 255, '1')
        threshold_img.save(f"{output_dir}/21_extreme_threshold_{i}_{threshold}.png")

# 9. Try to extract LSB (Least Significant Bit) - common in steganography
def extract_lsb(img_array):
    # Extract just the least significant bit
    lsb = img_array & 1
    # Scale to full intensity for visibility
    lsb = lsb * 255
    return lsb.astype(np.uint8)

lsb1 = extract_lsb(array1)
lsb2 = extract_lsb(array2)
lsb_xor = extract_lsb(xor_result)

Image.fromarray(lsb1).save(f"{output_dir}/22_lsb_image1.png")
Image.fromarray(lsb2).save(f"{output_dir}/23_lsb_image2.png")
Image.fromarray(lsb_xor).save(f"{output_dir}/24_lsb_xor.png")

print(f"Analysis complete. All output images saved to {output_dir} folder.")
print("Check all output images, especially the bit plane slices and LSB extractions, as hidden data often exists in these areas.")
