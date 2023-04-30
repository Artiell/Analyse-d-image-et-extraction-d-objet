import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt


def mainExec():

    # PARTIE 1 DU PROJET : Extraction des objets de l'image

    # definition de l'objet a trouver
    obj = cv2.imread('obj.jpg')
    cv2.imshow('obj',obj)

    #convertion en un autre espace de couleur
    convert_Obj = cv2.cvtColor(obj, cv2.COLOR_BGR2LAB)

    #chargement de l'image
    img = cv2.imread('puzzleImge.jpg')
    cv2.imshow('image',img)

    #convertion en un autre espace de couleur
    convert_Img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    #affichage en un autre espace de couleur
    cv2.imshow('ConvertImg',convert_Img)
    cv2.imshow('ConvertObj',convert_Obj)

    #histogramme de l'image et de l'objet

    #2 couche LAB
    objHisto = cv2.calcHist([convert_Obj], [1, 2], None, [256, 256], [0, 256, 0, 256])

    #normalisation des histogrammes
    Norm_objHisto = cv2.normalize(objHisto,objHisto,0,255,cv2.NORM_MINMAX)


    #1er itération de la backprojection

    # utilisation des cannaux a et b
    back_dst = cv2.calcBackProject([convert_Img],[1,2],Norm_objHisto,[0,256,0,256],1)

    #box filter
    back_dst_blur = cv2.boxFilter(back_dst,-1,(10,10),normalize=False)

    #binarisation de l'image après application du filtre moyenneur

    ret,res_thresh = cv2.threshold(back_dst_blur,20,255,0)


    # 2e itération de la backprojection

    # calcul du nouvel histogramme de l'objet qui est composé de l'image avec application du filtre
    objHisto2 = cv2.calcHist([convert_Img], [1, 2], res_thresh, [256, 256], [0, 256, 0, 256])

    #normalisation de l'histogramme
    Norm_objHisto2 = cv2.normalize(objHisto2,objHisto2,0,255,cv2.NORM_MINMAX)

    #application de la backprojection
    back_dst_2 = cv2.calcBackProject([convert_Img],[1,2],Norm_objHisto2,[0,256,0,256],1)

    # application d'un box filter
    back_dst_blur_2 = cv2.boxFilter(back_dst_2, -1, (7,7), normalize=False)

    #binarisation de l'image
    ret2,res_thresh_2 = cv2.threshold(back_dst_blur_2,20,255,0)



    #3e itération de la backprojection pour les finissions

    # calcul du nouvel histogramme de l'objet qui est composé de l'image avec application du filtre
    objHisto3 = cv2.calcHist([convert_Img], [1, 2], res_thresh_2, [256, 256], [0, 256, 0, 256])

    # normalisation de l'histogramme
    Norm_objHisto3 = cv2.normalize(objHisto3, objHisto3, 0, 255, cv2.NORM_MINMAX)

    # application de la backprojection
    back_dst_3 = cv2.calcBackProject([convert_Img], [1, 2], Norm_objHisto3, [0, 256, 0, 256], 1)

    # box filter
    back_dst_blur_3 = cv2.boxFilter(back_dst_3, -1, (7,7), normalize=False)

    # binarisation de l'image
    ret2, res_thresh_3 = cv2.threshold(back_dst_blur_3, 20, 255, 0)


    # 4e itération de la backprojection pour les finissions

    # calcul du nouvel histogramme de l'objet qui est composé de l'image avec application du filtre
    objHisto4 = cv2.calcHist([convert_Img], [1, 2], res_thresh_3, [256, 256], [0, 256, 0, 256])

    # normalisation de l'histogramme
    Norm_objHisto4 = cv2.normalize(objHisto4, objHisto4, 0, 255, cv2.NORM_MINMAX)

    # application de la backprojection
    back_dst_4 = cv2.calcBackProject([convert_Img], [1, 2], Norm_objHisto4, [0, 256, 0, 256], 1)

    # box filter
    back_dst_blur_4 = cv2.boxFilter(back_dst_4, -1, (7,7), normalize=False)


    # binarisation de l'image
    ret2, res_thresh_4 = cv2.threshold(back_dst_blur_4, 20, 255, 0)


    #en temps normal on devrait faire une dilatation pour combler les trous de l'image
    #mais dans notre cas on va faire une erosion car l'objet est noir sur fond blanc

    kernel = np.ones((4,4),np.uint8)
    erosion_res = cv2.erode(res_thresh_4,kernel,iterations = 2)


    #Affichage des différenets étapes de la backprojection

    # cv2.imshow('back_dst_1',back_dst)
    # cv2.imshow('res_step_1',res_thresh)

    # cv2.imshow('back_dst_2', back_dst_2)
    # cv2.imshow('res_step_2', res_thresh_2)
    #
    # cv2.imshow('back_dst_3', back_dst_3)
    # cv2.imshow('res_step_3', res_thresh_3)
    #
    # cv2.imshow('back_dst_4', back_dst_4)
    # cv2.imshow('res_step_4', res_thresh_4)

    cv2.imshow('erode_res', erosion_res)


    # PARTIE 2 DU PROJET : ISOLATION DES PIECES , DETECTION ET SIMPLIFICATION DES CONTOURS

    #petite triche on mets des pixels du contour de l'image en blanc pour séparer les pièces de la bordure

    res_Without_Border = erosion_res.copy()

    h,w = res_Without_Border.shape
    print(h,w)

    for i in range(10):
        res_Without_Border[i,::] = 255
        res_Without_Border[::,i] = 255

        res_Without_Border[-i, ::] = 255

    # on enlève la partie à droite avec le logo en bas a droite pour avoir un masque avec uniquelment les pièces
    for i in range(100):
        res_Without_Border[::, -i] = 255

    #on affiche sans les bordures et après avoir triché un petit peu
    cv2.imshow("res_without_border",res_Without_Border)


    #detection des contours par cv2 avec le filtre
    contours, hierarchy = cv2.findContours(res_Without_Border, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #on va filtrer les contours dans la liste obtenu, si on regarde dans "contours" qui est une liste de tableaux numpy qui les coordonnées des points des contours
    #on va considérer qu'un tableau de point plus petit que 150 point n'est pas une pièce de puzzle et donc qu'il s'agit d'un artefact de la backprojection

    real_cnt = list()
    for cnt in contours:
        if(len(cnt) >=150):
            real_cnt.append(cnt)

    print(len(real_cnt))

    #on affiche les contours exacts
    img_contours = cv2.drawContours(img.copy(), real_cnt, -1, (0, 255, 0), 2)
    cv2.imshow('Contours exacts', img_contours)


    #on va approximer les contours pour avoir des formes plus simples
    img_contours_simplified = img.copy()

    # Aproximation des contours
    for cnt in real_cnt:
        epsilon = 0.018 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        img_contours_simplified = cv2.drawContours(img_contours_simplified, [approx], -1, (0, 255, 0), 2)

    #on affiche les contours simplifiés
    cv2.imshow('Contours simplifies', img_contours_simplified)


    #on isole chaque pièce les unes des autres en dessinant des rectangles autou
    #on récupère l'image précédente avec les contours pour travailler dessus et dessiner des gros rectangles
    #on va chercher pour tous chaques contours les min et max en x et y de chaque pièces*

    res_with_rectangle = img.copy()

    maxX = 0
    minX = 20000
    maxY = 0
    minY = 20000

    for cnt in real_cnt:
        for pnts in cnt:

            if pnts[0,0] > maxY: maxY = pnts[0,0]
            if pnts[0,0] < minY: minY = pnts[0,0]

            if pnts[0,1] > maxX: maxX = pnts[0,1]
            if pnts[0,1] < minX: minX = pnts[0,1]

        #on dessine un rectangle autour de la pièce avec les min et max obtenu pour chaque pièce
        res_with_rectangle = cv2.rectangle(img=res_with_rectangle, pt1=(minY, minX), pt2=(maxY, maxX), color=(0, 255, 0), thickness=2)

        #on réinitialise les min et max pour la prochaine pièce
        maxX = 0
        minX = 20000
        maxY = 0
        minY = 20000

    #on affiche l'image avec les rectangles
    cv2.imshow("img_rectangle", res_with_rectangle)















    #fin du programme
    cv2.waitKey(0)
    cv2.destroyAllWindows()



mainExec()