import tensorflow as tf
import numpy as np
import itertools
import cv2
import time
from scipy import ndimage
from TianChiDateReader import TianChiDateReader

IMGSIZE=512
LANDMARK=26

Pixels = tf.constant(np.array([(x, y) for x in range(IMGSIZE) for y in range(IMGSIZE)], dtype=np.float32),shape=[IMGSIZE,IMGSIZE,2])
def GetHeatMap(Landmark):
    HalfSize = 8
    def Do(L):
        def DoIn(Point):
            return Pixels - Point
        Landmarks = tf.reverse(tf.reshape(L,[-1,2]),[-1])
        Landmarks = tf.clip_by_value(Landmarks,HalfSize,112 - 1 - HalfSize)
        Ret = 1 / (tf.norm(tf.map_fn(DoIn,Landmarks),axis = 3) + 1)
        Ret = tf.reshape(tf.reduce_max(Ret,0),[IMGSIZE,IMGSIZE,1])
        return Ret
    return tf.map_fn(Do,Landmark)


def PredictErr(GroudTruth,Predict):
    Gt = tf.reshape(GroudTruth,[-1,LANDMARK,2])
    Ot = tf.reshape(Predict,[-1,LANDMARK,2])

    def MeanErr(flt,Mix):
        Current,Gt = Mix
        MeanErr = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.subtract(Current, Gt) ** 2,1)))
        return MeanErr
    return tf.scan(fn=MeanErr,elems=[Ot,Gt],initializer=0.0)

Feed_dict = {}
Ret_dict = {}
def Layers():
    # MeanShape = tf.constant(Mshape)

    with tf.variable_scope('Stage1'):
        InputImage = tf.placeholder(tf.float32 ,[None ,IMGSIZE ,IMGSIZE ,1])
        GroundTruth = tf.placeholder(tf.float32 ,[None ,LANDMARK * 2])
        Visible=tf.placeholder(tf.int32,[None,LANDMARK*2])
        S1_isTrain = tf.placeholder(tf.bool)

        Feed_dict['InputImage'] = InputImage
        Feed_dict['GroundTruth'] = GroundTruth
        Feed_dict['S1_isTrain'] = S1_isTrain
        Feed_dict['Visible']=Visible

        S1_Conv1a = tf.layers.batch_normalization \
            (tf.layers.conv2d(InputImage ,64 ,3 ,1 ,padding='same' ,activation=tf.nn.relu
                             ,kernel_initializer=tf.glorot_uniform_initializer()) ,training=S1_isTrain)
        S1_Conv1b = tf.layers.batch_normalization \
            (tf.layers.conv2d(S1_Conv1a ,64 ,3 ,1 ,padding='same' ,activation=tf.nn.relu
                             ,kernel_initializer=tf.glorot_uniform_initializer()) ,training=S1_isTrain)
        S1_Pool1 = tf.layers.max_pooling2d(S1_Conv1b ,2 ,2 ,padding='same')

        S1_Conv2a = tf.layers.batch_normalization \
            (tf.layers.conv2d(S1_Pool1 ,128 ,3 ,1 ,padding='same' ,activation=tf.nn.relu
                             ,kernel_initializer=tf.glorot_uniform_initializer()) ,training=S1_isTrain)
        S1_Conv2b = tf.layers.batch_normalization \
            (tf.layers.conv2d(S1_Conv2a ,128 ,3 ,1 ,padding='same' ,activation=tf.nn.relu
                             ,kernel_initializer=tf.glorot_uniform_initializer()) ,training=S1_isTrain)
        S1_Pool2 = tf.layers.max_pooling2d(S1_Conv2b ,2 ,2 ,padding='same')

        S1_Conv3a = tf.layers.batch_normalization \
            (tf.layers.conv2d(S1_Pool2 ,256 ,3 ,1 ,padding='same' ,activation=tf.nn.relu
                             ,kernel_initializer=tf.glorot_uniform_initializer()) ,training=S1_isTrain)
        S1_Conv3b = tf.layers.batch_normalization \
            (tf.layers.conv2d(S1_Conv3a ,256 ,3 ,1 ,padding='same' ,activation=tf.nn.relu
                             ,kernel_initializer=tf.glorot_uniform_initializer()) ,training=S1_isTrain)
        S1_Pool3 = tf.layers.max_pooling2d(S1_Conv3b ,2 ,2 ,padding='same')

        S1_Conv4a = tf.layers.batch_normalization \
            (tf.layers.conv2d(S1_Pool3 ,512 ,3 ,1 ,padding='same' ,activation=tf.nn.relu
                             ,kernel_initializer=tf.glorot_uniform_initializer()) ,training=S1_isTrain)
        S1_Conv4b = tf.layers.batch_normalization \
            (tf.layers.conv2d(S1_Conv4a ,512 ,3 ,1 ,padding='same' ,activation=tf.nn.relu
                             ,kernel_initializer=tf.glorot_uniform_initializer()) ,training=S1_isTrain)
        S1_Pool4 = tf.layers.max_pooling2d(S1_Conv4b ,2 ,2 ,padding='same')

        S1_Pool4_Flat = tf.contrib.layers.flatten(S1_Pool4)
        S1_DropOut = tf.layers.dropout(S1_Pool4_Flat ,0.5 ,training=S1_isTrain)

        S1_Fc1 = tf.layers.batch_normalization \
            (tf.layers.dense(S1_DropOut ,256 ,activation=tf.nn.relu ,kernel_initializer=tf.glorot_uniform_initializer())
            ,training=S1_isTrain ,name = 'S1_Fc1')
        S1_Fc2 = tf.layers.dense(S1_Fc1 ,LANDMARK * 2)

        Mask=(Visible+Visible*Visible)/2
        S1_landmark=tf.multiply(S1_Fc2,Mask)
        S1_GROUNDTRUE=tf.multiply(GroundTruth,Mask)



        S1_Cost = tf.reduce_mean(PredictErr(S1_GROUNDTRUE ,S1_landmark))

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS ,'Stage1')):
            S1_Optimizer = tf.train.AdamOptimizer(0.001).minimize(S1_Cost ,var_list=tf.get_collection
                (tf.GraphKeys.TRAINABLE_VARIABLES ,"Stage1"))

        Ret_dict['S1_Landmark'] = S1_landmark
        Ret_dict['S1_Cost'] = S1_Cost
        Ret_dict['S1_Optimizer'] = S1_Optimizer

    with tf.variable_scope('Stage2'):
        S2_isTrain = tf.placeholder(tf.bool)
        Feed_dict['S2_isTrain'] = S2_isTrain
        #
        # S2_AffineParam = GetAffineParam(S1_Ret ,MeanShape)
        S2_InputImage = InputImage

        # S2_InputLandmark = AffineLandmark(S1_Ret ,S2_AffineParam)
        S2_InputHeatmap = GetHeatMap(S1_landmark)

        S2_Feature = tf.reshape(tf.layers.dense(S1_Fc1 ,int((IMGSIZE / 2) * (IMGSIZE / 2)) ,activation=tf.nn.relu
                                                ,kernel_initializer=tf.glorot_uniform_initializer())
                                ,(-1 ,int(IMGSIZE / 2) ,int(IMGSIZE / 2) ,1))
        S2_FeatureUpScale = tf.image.resize_images(S2_Feature ,(IMGSIZE ,IMGSIZE) ,1)

        S2_ConcatInput = tf.layers.batch_normalization \
            (tf.concat([S2_InputImage ,S2_InputHeatmap ,S2_FeatureUpScale] ,3) ,training=S2_isTrain)
        S2_Conv1a = tf.layers.batch_normalization \
            (tf.layers.conv2d(S2_ConcatInput ,64 ,3 ,1 ,padding='same' ,activation=tf.nn.relu
                             ,kernel_initializer=tf.glorot_uniform_initializer()) ,training=S2_isTrain)
        S2_Conv1b = tf.layers.batch_normalization \
            (tf.layers.conv2d(S2_Conv1a ,64 ,3 ,1 ,padding='same' ,activation=tf.nn.relu
                             ,kernel_initializer=tf.glorot_uniform_initializer()) ,training=S2_isTrain)
        S2_Pool1 = tf.layers.max_pooling2d(S2_Conv1b ,2 ,2 ,padding='same')

        S2_Conv2a = tf.layers.batch_normalization \
            (tf.layers.conv2d(S2_Pool1 ,128 ,3 ,1 ,padding='same' ,activation=tf.nn.relu
                             ,kernel_initializer=tf.glorot_uniform_initializer()) ,training=S2_isTrain)
        S2_Conv2b = tf.layers.batch_normalization \
            (tf.layers.conv2d(S2_Conv2a ,128 ,3 ,1 ,padding='same' ,activation=tf.nn.relu
                             ,kernel_initializer=tf.glorot_uniform_initializer()) ,training=S2_isTrain)
        S2_Pool2 = tf.layers.max_pooling2d(S2_Conv2b ,2 ,2 ,padding='same')

        S2_Conv3a = tf.layers.batch_normalization \
            (tf.layers.conv2d(S2_Pool2 ,256 ,3 ,1 ,padding='same' ,activation=tf.nn.relu
                             ,kernel_initializer=tf.glorot_uniform_initializer()) ,training=S2_isTrain)
        S2_Conv3b = tf.layers.batch_normalization \
            (tf.layers.conv2d(S2_Conv3a ,256 ,3 ,1 ,padding='same' ,activation=tf.nn.relu
                             ,kernel_initializer=tf.glorot_uniform_initializer()) ,training=S2_isTrain)
        S2_Pool3 = tf.layers.max_pooling2d(S2_Conv3b ,2 ,2 ,padding='same')

        S2_Conv4a = tf.layers.batch_normalization \
            (tf.layers.conv2d(S2_Pool3 ,512 ,3 ,1 ,padding='same' ,activation=tf.nn.relu
                             ,kernel_initializer=tf.glorot_uniform_initializer()) ,training=S2_isTrain)
        S2_Conv4b = tf.layers.batch_normalization \
            (tf.layers.conv2d(S2_Conv4a ,512 ,3 ,1 ,padding='same' ,activation=tf.nn.relu
                             ,kernel_initializer=tf.glorot_uniform_initializer()) ,training=S2_isTrain)
        S2_Pool4 = tf.layers.max_pooling2d(S2_Conv4b ,2 ,2 ,padding='same')

        S2_Pool4_Flat = tf.contrib.layers.flatten(S2_Pool4)
        S2_DropOut = tf.layers.dropout(S2_Pool4_Flat ,0.5 ,training=S2_isTrain)

        S2_Fc1 = tf.layers.batch_normalization \
            (tf.layers.dense(S2_DropOut ,256 ,activation=tf.nn.relu ,kernel_initializer=tf.glorot_uniform_initializer())
            ,training=S2_isTrain)
        S2_Fc2 = tf.layers.dense(S2_Fc1 ,LANDMARK * 2)

        d_landmark=tf.multiply(S2_Fc2,Mask)


        S2_groundtrue=GroundTruth-d_landmark



        # S2_Ret = AffineLandmark(S2_Fc2 + S2_InputLandmark ,S2_AffineParam ,isInv=True)
        S2_Cost = tf.reduce_mean(PredictErr(S2_groundtrue ,d_landmark))

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS ,'Stage2')):
            S2_Optimizer = tf.train.AdamOptimizer(0.001).minimize(S2_Cost ,var_list=tf.get_collection
                (tf.GraphKeys.TRAINABLE_VARIABLES ,"Stage2"))

        Ret_dict['d_landmark'] = d_landmark
        Ret_dict['S2_Cost'] = S2_Cost
        Ret_dict['S2_Optimizer'] = S2_Optimizer

        # Ret_dict['S2_InputImage'] = S2_InputImage
        # Ret_dict['S2_InputLandmark'] = S2_InputLandmark
        # Ret_dict['S2_InputHeatmap'] = S2_InputHeatmap
        # Ret_dict['S2_FeatureUpScale'] = S2_FeatureUpScale
    return


Layers()
STAGE = 2
rt = TianChiDateReader("../train_modified", train_size=0.95)
val_images, val_landmarks, val_visibilities = rt.get_validation_data()

with tf.Session() as Sess:
    Saver = tf.train.Saver()
    Writer = tf.summary.FileWriter("logs/", Sess.graph)
    if STAGE == 0:
        Sess.run(tf.global_variables_initializer())
    else:
        Saver.restore(Sess, './Model/Model')
        print('Model Read Over!')

    # Landmark68Test(MeanShape, ImageMean, ImageStd, Sess)
    Count = 0
    for w in range(100):
        # RandomIdx = np.random.choice(I.shape[0], 64, False)
        for n in range(num_batch):
            images, landmarks, visibilities=rt.get_train_data(batch_size=32)
            if STAGE == 1 or STAGE == 0:
                Sess.run(Ret_dict['S1_Optimizer'],
                     {Feed_dict['InputImage']: images, Feed_dict['GroundTruth']: landmarks,Feed_dict['Visible']:visibilities,
                      Feed_dict['S1_isTrain']: True, Feed_dict['S2_isTrain']: False})
            else:
                Sess.run(Ret_dict['S2_Optimizer'],
                     {Feed_dict['InputImage']: images, Feed_dict['GroundTruth']: landmarks,Feed_dict['Visible']:visibilities,
                      Feed_dict['S1_isTrain']: False, Feed_dict['S2_isTrain']: True})
            Count += 1

            if Count%100==0:
                if STAGE == 1 or STAGE == 0:
                    cost=Ret_dict['S1_Cost']
                else:
                    cost=Ret_dict['S2_Cost']
                ValErr = Sess.run(cost,
                        {Feed_dict['InputImage']: val_images, Feed_dict['GroundTruth']: val_landmarks,Feed_dict['Visible']:val_visibilities,
                         Feed_dict['S1_isTrain']: False,Feed_dict['S2_isTrain']: False})
                BatchErr = Sess.run(cost,
                                        {Feed_dict['InputImage']: images, Feed_dict['GroundTruth']: landmarks,Feed_dict['Visible']:visibilities,
                                         Feed_dict['S1_isTrain']: False, Feed_dict['S2_isTrain']: False})
                print( 'step:',Count, 'ValErr:', ValErr, ' BatchErr:', BatchErr)

            if Count%1000==0:
                Saver.save(Sess, './Model/Model')
        # if Count % 256 == 0:
        #     TestErr = 0
        #     BatchErr = 0
        #
        #     if STAGE == 1 or STAGE == 0:
        #         TestErr = Sess.run(Ret_dict['S1_Cost'], {Feed_dict['InputImage']: Ti, Feed_dict['GroundTruth']: Tg,
        #                                                      Feed_dict['S1_isTrain']: False,
        #                                                      Feed_dict['S2_isTrain']: False})
        #         BatchErr = Sess.run(Ret_dict['S1_Cost'],
        #                                 {Feed_dict['InputImage']: I[RandomIdx], Feed_dict['GroundTruth']: G[RandomIdx],
        #                                  Feed_dict['S1_isTrain']: False, Feed_dict['S2_isTrain']: False})
        #     else:
                    # Landmark,Img,HeatMap,FeatureUpScale =
                    # Sess.run([Ret_dict['S2_InputLandmark'],Ret_dict['S2_InputImage'],Ret_dict['S2_InputHeatmap'],Ret_dict['S2_FeatureUpScale']],{Feed_dict['InputImage']:I[RandomIdx],Feed_dict['GroundTruth']:G[RandomIdx],Feed_dict['S1_isTrain']:False,Feed_dict['S2_isTrain']:False})
                    # for i in range(64):
                    #    TestImage = np.zeros([112,112,1])
                    #    for p in range(68):
                    #        cv2.circle(TestImage,(int(Landmark[i][p *
                    #        2]),int(Landmark[i][p * 2 + 1])),1,(255),-1)

                    #    cv2.imshow('Landmark',TestImage)
                    #    cv2.imshow('Image',Img[i])
                    #    cv2.imshow('HeatMap',HeatMap[i])
                    #    cv2.imshow('FeatureUpScale',FeatureUpScale[i])
                    #    cv2.waitKey(-1)




