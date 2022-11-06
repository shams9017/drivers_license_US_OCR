DL Extractor Documentation

How to Run:
1. Install missing packages using pip install. You have to install them one by one. For tesseract, you will have to download and install it manually.
2. Clicking run should compile the code and tun it. Results will be shown in the terminal. You can also debug if you need in your IDE.
Limitations: 
1. Images must be in a higher resolution and not pixelated.
2. Unable to yield perfect results for all images. At times, Tesseract can be unreliable. Use of denoise and other pre-processing is needed in a trial and error fashion.
3. Needs to manually add preprocessing before text extraction.
4. Photos of some images may need to be cropped to yield better results. If an image has extra space other than content, then they might need to be cropped as well.
5. Extraction is done based on assumption. For instance, it is assumed the first date on the DL is the expiry date. So, based on that assumption all the other data points are derived.
Comments:
You can add as many pre-processing as you want. But unnecessary ones will not yield results or it can also be impossible to find the correct pre-processing for an image which can lead to no acceptable results. Each image has a certain combination of pre-processing. I have added some of these combinations on the test images. Make sure to add the correct path to the images and play around.   
