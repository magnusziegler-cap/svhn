import numpy as np
import math

def rotate(X, max_rotation_angle_degrees=15, nearest_neighbour=True):
    """Randomly rotate the image, about its centre

    Args:
        X (ndarray): image matrix in [CHW] or [HW] dimension order
        max_rotation_angle_degrees (int, optional): maximum rotation angle. Defaults to 15.
        nearest_neighbour (bool, optional): how to fill the pixels. If false, basic interpolation used. Defaults to True.

    Returns:
        _type_: ndarray
    """
    # rotate from centre, find new location pixel by pixel.
    if X.ndim==2:
        img_x, img_y = X.shape
        max_side_length = int(np.sqrt(img_x**2 + img_y**2))
        rshape = [max_side_length, max_side_length]
    elif X.ndim==3:
        img_c, img_x, img_y = X.shape
        rshape.append(img_c)
        rshape = np.roll(rshape, -1) #move C to end
        max_side_length = int(np.sqrt(img_x**2 + img_y**2))

     #45 degrees is bad, could minimize this by moving it down to after angle is chosen.
    # pre-create output image with max possible size, crop later
    rotated = np.zeros(shape=rshape)

    #centre coords, rotate around these
    cx = max_side_length//2
    cy = max_side_length//2
    
    #with this randomly chosen angle within the bounds
    angle_degrees = np.random.choice([-1,1])*np.random.random(1)*max_rotation_angle_degrees
    angle_rad = math.radians(angle_degrees)

    # roll through the output-size image, for each pixel, try to find valid one in input image
    for j in range(0, max_side_length):
        for i in range(0, max_side_length):
            # apply rotation
            x = (i-cx)*np.cos(angle_rad) - (j-cy)*np.sin(angle_rad)
            y = (i-cx)*np.sin(angle_rad) + (j-cy)*np.cos(angle_rad)

            # shift
            x += cx
            y += cy

            # Nearest Neighbour
            if nearest_neighbour:
                x = round(x)
                y = round(y)

                if (x>=0 and x<img_x) and (y>=0 and y<img_y):
                    rotated[i,j,...] = X[x,y,...]

            else:
                # quick-interp
                x_low = int(np.floor(x))
                x_high = int(np.ceil(x))
                y_low = int(np.floor(y))
                y_high = int(np.ceil(y))
                
                #prevent case where they are the same?
                if x_low == x_high:
                    x_high +=1
                if y_low == y_high:
                    y_high +=1

                if (x_low>=0 and x_high<img_x) and (y_low>=0 and y_high<img_y):
                    d_lh = np.sqrt((x_low-x_high)**2 + (y_low-y_high)**2)
                    d_ln = np.sqrt((x_low-x)**2 + (y_low-y)**2)
                    d_nh = d_lh - d_ln
                    
                    rotated[i,j,...] = (X[x_low,y_low,...]*d_ln + X[x_high,y_high,...]*d_nh) / d_lh

    #crop output
    output = rotated[0:img_x, 0:img_y, ...]
    
    if output.ndim==3:
        output = np.transpose(output, axes=(2,0,1))

    return output

def gaussian_blur(X, sigma=0.05, width=5):
    """Apply Gaussian Blur to Image

    Args:
        X (ndarray): image matrix in [CHW] or [HW] dimension order
        sigma (float, optional): sigma parameter from gaussian distribution. Defaults to 0.05.
        width (int, optional): kernel width. Defaults to 5.

    Returns:
        _type_: ndarray
    """
    a = np.linspace(-np.floor(width/2), np.floor(width/2), width) #linear space from [-width/2,width/2]
    g = (1/np.sqrt(2*np.pi)) * np.exp(0.5*np.square(a))/np.square(sigma) #gaussian function over axis a
    kernel = np.outer(g,g) # make it a square
    kernel = kernel / np.sum(kernel) #normalize
    
    output = conv2d(X, kernel, padding=int(np.floor(width/2)), stride=(1,1)) #perform conv

    return output

def normalize(X):
    """min_max normalization

    Args:
        X (ndarray): image matrix in [CHW] or [HW] dimension order

    Returns:
        _type_: ndarray
    """
    #min-max scale
    # element-wise divide by max per channel
    output = X
    
    # todo: generalize for better multidimensional support
    if output.ndim==3:
        for channel in range(X.shape[0]):
            output[channel,...] = np.divide(X[channel,...], X[channel,...].max())
    else:
        output = np.divide(X, X.max())
    return output

def znormalize(X):
    """z-normalization

    Args:
        X (ndarray): image matrix in [CHW] or [HW] dimension order

    Returns:
        _type_: ndarray
    """
    #z-normalization
    
    ## z = (x-mean)/std
    output = X
    # todo: generalize for better multidimensional support
    if output.ndim==3:
        for channel in range(0,X.shape[0]):
            output[channel,...] = (X[channel,...] - np.mean(X[channel,...])) / np.std(X[channel,...])
    else:
        output = (X - np.mean(X)) / np.std(X)

    return output 

def shift(X, maximum_shift=(10,), axis=None):
    """Shift (Translate) image LR/UD

    Args:
        X (ndarray): image matrix in [CHW] or [HW] dimension order
        maximum_shift (int/iterable, optional): maximum allowable shift in pixels, Defaults to (10).
        axis (tuple, optional): axes to apply on. Defaults to final two dimensions of input array.

    Returns:
        _type_: ndarray
    """
    if axis==None:
        axis = np.arange(0, X.ndim)
        axis = axis[-2:] #keep final two
    if len(maximum_shift) != axis.size:
        maximum_shift = np.repeat(maximum_shift,len(axis)) 

    #Randomly Shift the image up,down, left, or right
    assert len(maximum_shift)==len(axis)

    output = X
    for s, d in zip(maximum_shift,axis):
        # max shift is positive-and-negative
        output = np.roll(output,
                    shift = np.random.randint(low=-s, high=s),
                    axis = d)


    return output

def conv2d(X, kernel, padding=None, stride=(1,1)):
    """2D convolution

    Args:
        X (ndarray): image matrix in [CHW] or [HW] dimension order
        kernel (ndarray): kernel
        padding (int, optional): padding to be applied. Defaults to None.
        stride (tuple, optional): stride value for kernel. Defaults to (1,1).

    Returns:
        _type_: ndarray
    """
    #sizes
    kx = kernel.shape[0]
    ky = kernel.shape[1]

    sx = stride[0]
    sy = stride[1]

    img_x, img_y = X.shape[-2:]
    
    #output sizes and img
    ox = int((np.floor(img_x - kx + 2*padding) / stride[0]) + 1)
    oy = int((np.floor(img_y - ky + 2*padding) / stride[1]) + 1)

    if padding == None:
        padding = int(np.ceil(max(kx,ky)/2))

    #todo: better dimension handling and flexibility. this is ugly
    if X.ndim==2:
        
        output = np.zeros(shape=(ox,oy))
        padded_image = np.zeros(shape=((img_x + 2*padding),(img_y + 2*padding))) #even padding assumed

    elif X.ndim==3:
        
        img_c = X.shape[0]

        output = np.zeros(shape=(img_c,ox,oy))
        padded_image = np.zeros(shape=(img_c,(img_x + 2*padding),(img_y + 2*padding))) #even padding assumed

    padded_image[padding:img_x+padding, padding:img_y+padding, ...] = X

    for y in range(padding, img_y, sy): #roll through y from inner pad border to orig image border
        for x in range(padding, img_x, sx): # roll through x from inner pad border to orig image border
            if X.ndim==3:
                for c in range(0, img_c): #process channels if exist
                    output[c,x,y] = (kernel * padded_image[c,(x-kx//2):(x+kx//2+1),(y-ky//2):(y+ky//2+1)]).sum()
            else:
                output[x,y] = (kernel * padded_image[(x-kx//2):(x+kx//2+1),(y-ky//2):(y+ky//2+1)]).sum()

    return output


def grayscale(X, R=0.35, G=0.6, B=0.15):
    """Convert RGB image to grayscale

    Args:
        X (ndarray): image matrix in [CHW] 
        R (float, optional): scalar RGB-to-gray mix proportion. Defaults to 0.35.
        G (float, optional): scalar RGB-to-gray mix proportion. Defaults to 0.6.
        B (float, optional): scalar RGB-to-gray mix proportion. Defaults to 0.15.

    Returns:
        _type_: ndarray
    """
    #Convert RGB to Grayscale. assumed CHW dim order
    output = np.zeros(shape=(1,X.shape[1],X.shape[2]))

    # gray is roughly = 0.35*R + 0.6*G + 0.15*B
    # but it probably doesn't matter so much
    output = X[0,...]*R + X[1,...]*G + X[2,...]*B

    return output

def noise(X, sigma=0.15, probability=0.5):
    """ Adds noise to image.
    sigma defines the normal distribution of the noise (+/-).
    probability defines how likely the pixel is to get noise.
    
    Args:
        sigma (float, optional): sigma parameter from gaussian distribution. Defaults to 0.05.
        width (int, optional): kernel width. Defaults to 5.

    Returns:
        _type_: ndarray
    """
    
    # normally distributed noise about 0 (to account for direction)
    X += X * (np.random.random(size=X.size) > probability) * np.random.normal(loc=0, scale=sigma, size=X.size)

def compose(*funcs, probabilities, verbose=False):
    """
    Stack augmentations and compose them into a single function.
    Only add the function if it is under the probability threshold.
    Probability threshold defined per function.
    """
    def apply(x):
        #todo: kwargs?
        for func, probability in zip(funcs, probabilities):
            if np.random.random() <= probability:
                if verbose:
                    print(f'Applied Transformation Function: {func.__name__}')
                x = func(x)
        return x

    return apply
