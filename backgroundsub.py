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
oceanVideo = cv.VideoCapture("praia.mp4")

# Guarda o frame da webcam selecionado, para ser usado como referência
ret, bgReference = video.read()

# Variável que controla se um novo frame de plano de fundo deve ser selecionado
takeBgImage = 0

while(1):
    # Lê o frame da webcam 
    ret,img = video.read()
    # Lê o frame do vídeo de fundo
    ret2, bg = oceanVideo.read()
    
    # Redimensiona o fundo para o mesmo tamanho da imagem da webcam
    if bg is not None:
        bg = resize(bg,bgReference)
        
    # Se a tecla 'e' for pressionada, captura o frame da webcam como referência
    if takeBgImage == 0:
        bgReference = img
        
    # Calcula a diferença entre o frame da webcam e o frame de referência
    # Cria uma máscara de diferença
    diff1 = cv.subtract(img,bgReference)
    diff2 = cv.subtract(bgReference,img)
    
    # Soma as diferenças calculadas
    diff = diff1 + diff2
    # Verifica a diferênça entre os pixels, caso seja menor que 25, ele é igual a 0, diminuindo possíveis ruidos da imagem 
    diff[abs(diff) < 25.0] = 0
    
    # Exibe imagem da diferença calculada
    cv.imshow("diff1", diff)
    
    # Verifica novamente cada pixel para diminuição de ruído
    # Converte a imagem para escala de cinza utilizando a função cvtColor
    gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
    # Filtra os pixels com valor absoluto menor que 10, retirando valores de ruído
    gray[np.abs(gray) < 10] = 0
    fgMask = gray
    
    #Cria um kernel, 3x3, vai ser usado para aplicar oqperações morfológicas para perfomar a erosão e dilatação na máscara 
    kernel = np.ones((3,3), np.uint8)
    
    #Erosão é uma operação morfológica que reduz o tamanho de regiões brancas na imagem
    fgMask = cv.erode(fgMask, kernel, iterations=2)
    #Dilatação é uma operação morfológica que aumenta o tamanho de regiões brancas na imagem
    fgMask = cv.dilate(fgMask, kernel, iterations=2)
    #Essas operações uma seguida da outra, removem ruiídos menores, deixam as bordas mais suaves e preenchem buracos
    
    # Aplica limiarização na máscara
    # Limiarização é uma técnica de segmentação de imagens que separa os pixels em duas categorias, preto e branco
    # 0 = preto, 255 = branco / backgroud e foreground
    fgMask[fgMask > 5] = 255
    
    # Mostra a máscara de diferença
    cv.imshow("Foreground Mask", fgMask)
    
    # Inverte a máscara de diferença
    fgMask_inv = cv.bitwise_not(fgMask).astype(np.uint8)
    # Converte a máscara de diferença para o tipo uint8
    fgMask = np.uint8(fgMask)
    # Converte a máscara de diferença invertida para o tipo uint8
    fgMask_inv = np.uint8(fgMask_inv)
    
    # Retira os objetos que não pertencem ao fundo e armazenam eles
    fgImage = cv.bitwise_and(img, img, mask=fgMask)
    # Retira os objetos que pertencem ao fundo e armazenam eles
    bgImage = cv.bitwise_and(bg, bg, mask=fgMask_inv)
    #Combina os dois objetos e cria a imagem final com o fundo removido
    bgSub = cv.add(bgImage,fgImage)
    
    cv.imshow("Background Removed",bgSub)
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
