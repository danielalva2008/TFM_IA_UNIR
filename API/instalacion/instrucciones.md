1. **Crear el archivo del servicio systemd:**
   - Ejecuta el siguiente comando para crear un nuevo archivo de servicio systemd:
     ```bash
     sudo vi /etc/systemd/system/flask_app.service
     ```

2. **Editar el archivo del servicio:**
   - Dentro del editor, copia y pega el siguiente contenido en el archivo `flask_app.service`:
     ```plaintext
     [Unit]
     Description=Tu Aplicación Flask
     After=network.target

     [Service]
     User=root
     Group=root
     WorkingDirectory=/root/api
     Environment="FLASK_APP=app.py"
     ExecStart=/usr/bin/python3.11 -m flask run --host=0.0.0.0 --port=80
     Restart=always
     RestartSec=3

     [Install]
     WantedBy=multi-user.target
     ```
   - Asegúrate de ajustar `Description`, `WorkingDirectory` y el nombre del archivo de la aplicación Flask según corresponda.

3. **Guardar y salir del editor:**
   - En `vi`, presiona `Esc`, escribe `:wq` y presiona `Enter`.

4. **Recargar systemd y activar el servicio:**
   - Ejecuta los siguientes comandos:
     ```bash
     sudo systemctl daemon-reload
     sudo systemctl start flask_app
     ```

5. **Verificar el estado del servicio:**
   - Puedes verificar si el servicio se está ejecutando correctamente con el siguiente comando:
     ```bash
     sudo systemctl status flask_app
     ```

6. **Habilitar el inicio automático en el arranque:**
   - Si deseas que el servicio se inicie automáticamente en el arranque del sistema, ejecuta:
     ```bash
     sudo systemctl enable flask_app
     ```
