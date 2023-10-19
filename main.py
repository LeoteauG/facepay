import cv2

# Aquí coloca la ruta de la imagen con la cara a detectar
imagenRuta = 'imagen.jpeg'

# Cargar el modelo pre-entrenado para detección facial
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Cargar la imagen en la que deseas realizar el reconocimiento facial
image = cv2.imread( imagenRuta )

# Convertir la imagen a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detectar caras en la imagen
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Dibujar un rectángulo alrededor de cada cara detectada
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Mostrar la imagen con las caras marcadas
cv2.imshow("Reconocimiento Facial", image)

# Esperar a que el usuario presione una tecla y luego cerrar la ventana
cv2.waitKey(0)
cv2.destroyAllWindows()
