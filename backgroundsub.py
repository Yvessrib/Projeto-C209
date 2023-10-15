import cv2 as cv #OpenCV
import numpy as np #NumPy

# Função que redimensiona o fundo para o mesmo tamanho da imagem da webcam
def resize(dst,img):
    width = img.shape[1]
    height = img.shape[0]
    dim = (width, height)
    resied = cv.resize(dst,dim,interpolation=cv.INTER_AREA)
    return resied

# Função que captura a imagem da camera
video = cv.VideoCapture(0, cv.CAP_DSHOW)

# Vídeo que vai ser inserido no plano de fundo
praiaVideo = cv.VideoCapture("praia.mp4")

# Guarda o frame da webcam selecionado, para ser usado como referência
frame, backGroundReference = video.read()

# Variável que controla se um novo frame de plano de fundo deve ser selecionado
takeBgImage = 0

while(1):
    # Lê o frame da webcam 
    frame,img = video.read()
    # Lê o frame do vídeo de fundo
    frame2, backGround = praiaVideo.read()
    
    # Redimensiona o fundo para o mesmo tamanho da imagem da webcam
    if backGround is not None:
        backGround = resize(backGround,backGroundReference)
        
    # Se a tecla 'e' for pressionada, captura o frame da webcam como referência
    if takeBgImage == 0:
        backGroundReference = img
        
    # Calcula a diferença entre o frame da webcam e o frame de referência
    # Cria uma máscara de diferença
    dif1 = cv.subtract(img,backGroundReference)
    dif2 = cv.subtract(backGroundReference,img)
    
    # Soma as diferenças calculadas
    dif = dif1 + dif2
    # Verifica a diferênça entre os pixels, caso seja menor que 25, ele é igual a 0, diminuindo possíveis ruidos da imagem 
    dif[abs(dif) < 25.0] = 0
    
    # Exibe imagem da diferença calculada
    cv.imshow("subtração", dif)
    
    # Verifica novamente cada pixel para diminuição de ruído
    # Converte a imagem para escala de cinza utilizando a função cvtColor
    cinza = cv.cvtColor(dif, cv.COLOR_BGR2GRAY)
    # Filtra os pixels com valor absoluto menor que 10, retirando valores de ruído
    cinza[np.abs(cinza) < 10] = 0
    foregroundMask = cinza
    
    #Cria um kernel, 3x3, vai ser usado para aplicar operações morfológicas para perfomar a erosão e dilatação na máscara 
    kernel = np.ones((3,3), np.uint8)
    
    #Erosão é uma operação morfológica que reduz o tamanho de regiões brancas na imagem
    foregroundMask = cv.erode(foregroundMask, kernel, iterations=2)
    #Dilatação é uma operação morfológica que aumenta o tamanho de regiões brancas na imagem
    foregroundMask = cv.dilate(foregroundMask, kernel, iterations=2)
    #Essas operações uma seguida da outra, removem ruídos menores, deixam as bordas mais suaves e preenchem buracos
    
    # Aplica limiarização na máscara
    # Limiarização é uma técnica de segmentação de imagens que separa os pixels em duas categorias, preto e branco
    # 0 = preto, 255 = branco / backgroud e foreground
    foregroundMask[foregroundMask > 5] = 255 
    
    # Mostra a máscara de diferença
    cv.imshow("Foreground Mask", foregroundMask)
    
    # Inverte a máscara de diferença
    foregroundMask_inv = cv.bitwise_not(foregroundMask).astype(np.uint8)
    
    # Converte a máscara de diferença para o tipo uint8
    foregroundMask = np.uint8(foregroundMask)
    
    # Retira os objetos que não pertencem ao fundo e armazenam eles
    foregroundImage = cv.bitwise_and(img, img, mask=foregroundMask)
    # Retira os objetos que pertencem ao fundo e armazenam eles
    backgroundImage = cv.bitwise_and(backGround, backGround, mask=foregroundMask_inv)
    #Combina os dois objetos e cria a imagem final com o fundo removido
    backgroundSub = cv.add(backgroundImage,foregroundImage)
    
    cv.imshow("Background Removed",backgroundSub)
    cv.imshow("Original",img)
    
    # Verifica se alguma tecla foi pressionada
    key = cv.waitKey(5) & 0xFF
    # Se a tecla 'q' for pressionada, fecha o programa
    if ord('q') == key:
        break
    # Se a tecla 'e' for pressionada, captura o frame da webcam como referência
    elif ord('e') == key:
        takeBgImage = 1
        print("Background Captured")
    # Se a tecla 'r' for pressionada, reseta o frame de referência
    elif ord('r') == key:
        takeBgImage = 0
        print("Background Reset")
        
# Fecha todas as janelas
cv.destroyAllWindows()
# Fecha a webcam
video.release()
