
img_file = "filename"; %Replace filename with this vairable.

%Original Image load
Ioriginal = im2double(imread(img_file));
figure, imshow(Ioriginal)
title('Original Image')


% Blurring the images
PSF = fspecial('motion',21,11); % Motion Blur , 21 pixel , 11 degrees
Idouble = im2double(Ioriginal);
blurred = imfilter(Idouble,PSF,'conv','circular');
figure, imshow(blurred)
title('Blurred Image')

noise_mean = 0;
noise_var = 0.0001;
blurred_noisy = imnoise(blurred,'gaussian',noise_mean,noise_var);
figure,imshow(blurred_noisy)
title('Blurred and Noisy Image')

% Accounting for Noise
signal_var = var(Idouble(:));
NSR = noise_var / signal_var;
wnr3 = deconvwnr(blurred_noisy,PSF,NSR);
figure,imshow(wnr3)
title('Restoration of Blurred Noisy Image (Estimated NSR)')