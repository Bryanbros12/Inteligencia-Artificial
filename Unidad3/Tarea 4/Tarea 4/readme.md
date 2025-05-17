El programa que se presenta a continuación toma como base la base de conocimiento que se presento en los anteriores trabajos y se decidió por tomar como datos de entrada el estado de ánimo, la actividad que el usuario esta desarrollando, así como el genero de su preferencia y la compara con las canciones que este mas ha escuchado para hacer una comparativa de que tanto se relaciona con sus gustos.


CONEXIÓN_SPOTIFY.PY
Empezemos por como se conecta la api a una cuenta Spotify, para la implenetacion de esta api hay que registrarse en Spotify for developers y generar una llave de cliente, una llave secreta y un rediretorio uri como se muestra en el código, al tener dichas credenciales ala mano y ingresarlas nos da acceso a cierto numero de datos de Spotify ya que al ser una API gratuita esta un poco limitada y por eso no se llegaron a usar todas las reglas previamente definidas en el trabajo anterior.
 





PREFERENCIAS_USUARIO.PY
En esta sección del proyecto se toman en cuenta las preferencias que desea el usuario y la situación en la que se encuentra asi como se accede alas canciones mas escuchadas por el en el ultimo año con el comando “sp” el cual hace la solicitud a Spotify para recaudar los datos de la canciones mas escuchadas en este caso el top 5, devolviendo el nombre de la canción, artista y el link para ir ala canción.
 


MAIN.PY
En este parte principal del código es donde se ejecuta la recomendación de la música tomando en cuenta los valores ingresados mencionados previamente donde estos resultados se guardan en una consulta y en el método de recomendar_canciones devuelve el resultado en forma de print con las 5 canciones recomendadas para los datos insertados y debajo de este resultado de recomendación imprime las canciones mas escuchadas de tu perfil el top 5 que ya habíamos mencionado para que hagas un poco la comparativa de lo que recomienda y lo que tu escuchas.
 

INTERFAZ.PY
Y por ultimo en esta sección del proyecto es donde se lleva acabo la interfaz del programa y manda a llamar a todas las demás funciones necesarias tomando los datos de los campos de texto y imprimiéndolos de forma estética en un TEXTAREA y usando un botón para accionar la opción de “ejecutar_recomendacion” entregando el resultado que se va mostrar en las ultimas capturas.
