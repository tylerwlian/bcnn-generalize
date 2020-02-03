# Adapted from PowerOfCreation (GitHub)

from keras.models import Model, load_model
from keras.preprocessing import image
from keras.layers.core import Lambda
import keras.backend as K
import tensorflow as tf
import numpy as np
import cv2
import argparse


def get_flags():
    parser = argparse.ArgumentParser(description='Define experimental parameters')
    parser.add_argument('--img_name', type=str, default='', help='Image file name')
    parser.add_argument('--saved_model', type=str, help='Specify path to saved model')
    parser.add_argument('--model_type', type=str, help='Model class (e.g., dn, mcdo, mcbn)')
    parser.add_argument('--data_str', type=str, help='Train data source (nih, stf, mit)')
    parser.add_argument('--runs', type=int, default=1, help='Number of runs, if BCNN')
    parser.add_argument('--bbox', nargs='+', type=float, help='NIH coordinates (x,y,w,h)')
    return parser.parse_args()


def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))


def target_category_loss_output_shape(input_shape):
    return input_shape


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def process_image(img):
    # normalize and scale image by ImageNet
    x = image.img_to_array(img)
    x /= 255
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    x = (x - mean) / std
    return np.expand_dims(x, axis=0)


def _compute_gradients(tensor, var_list):
    grads = tf.gradients(tensor, var_list)
    return [grad if grad is not None else tf.zeros_like(var) for var, grad in zip(var_list, grads)]


def grad_cam(input_model, image, category_index, layer_name, T=1):
    nb_classes = 7
    target_layer = lambda x: target_category_loss(x, category_index, nb_classes)
    x = Lambda(target_layer, output_shape=target_category_loss_output_shape)(input_model.output)
    model = Model(inputs=input_model.input, outputs=x)
    # model.summary()

    loss = K.sum(model.output)
    conv_output = [l for l in model.layers if l.name == layer_name][0].output
    grads = normalize(_compute_gradients(loss, [conv_output])[0])
    gradient_function = K.function([model.input], [conv_output, grads])

    preprocessed_input = process_image(image)

    # generate first CAM
    output, grads_val = gradient_function([preprocessed_input])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.ones(output.shape[0:2], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    # if T > 1, superimpose CAMs
    for _ in range(T - 1):
        output, grads_val = gradient_function([preprocessed_input])
        output, grads_val = output[0, :], grads_val[0, :, :, :]

        weights = np.mean(grads_val, axis=(0, 1))

        for i, w in enumerate(weights):
            cam += w * output[:, :, i]

    cam = cv2.resize(cam, (224, 224))
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    cam = 0.4 * np.float32(cam) + 0.6 * np.float32(image)
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam), heatmap


def main():
    FLAGS = get_flags()
    print('User input parameters:', FLAGS)

    IMG_PATH = './data/NIH/images/test/'
    img = image.load_img(IMG_PATH + FLAGS.img_name, target_size=(224, 224))
    # img = image.load_img('./data/Stanford/CheXpert-v1.0-small/train/patient48271/study1/view1_frontal.jpg', target_size=(224, 224))

    model = load_model(FLAGS.saved_model)

    if FLAGS.model_type == 'dn':
        conv_layer = 'conv5_block16_2_conv'  # DenseNet architecture
    else:
        conv_layer = 'block5_conv4'  # VGG-19 architecture

        print('Creating CAM...')
    predicted_class = 5  # pneumonia index
    cam, heatmap = grad_cam(model, img, predicted_class, conv_layer, FLAGS.runs)

    orig_img = image.load_img(IMG_PATH + FLAGS.img_name)
    # orig_img = image.load_img('./data/Stanford/CheXpert-v1.0-small/train/patient48271/study1/view1_frontal.jpg')
    heatmap = cv2.resize(heatmap, (orig_img.size[0], orig_img.size[1]))

    cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    cam = 0.4 * np.float32(cam) + 0.6 * np.float32(orig_img)
    cam = 255 * cam / np.max(cam)
    cam = np.uint8(cam)

    # superimposes bounding box
    if FLAGS.bbox is not None:
        print('Drawing bounding box...')
        x, y, w, h = FLAGS.bbox
        cam = cv2.rectangle(cam, (int(x), int(y)), (int(x + w), int(y + h)),
                            color=(0, 0, 255), thickness=2)

    filename = FLAGS.model_type + '_' + FLAGS.data_str + '_' + FLAGS.img_name + '.jpg'
    cv2.imwrite(filename, cam)

    print('Done. Image saved as:', filename)


if __name__ == '__main__':
    main()
