const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const path = require('path');

// Создаем Express приложение
const app = express();
const server = http.createServer(app);
const io = socketIo(server);

// Конфигурация
const PORT = process.env.PORT || 3000;

// Раздаем статические файлы из папки public
app.use(express.static(path.join(__dirname, 'public')));

// Обработка подключений Socket.io
io.on('connection', (socket) => {
    console.log(`Новое подключение: ${socket.id}`);
    
    // Уведомляем всех о новом пользователе
    socket.broadcast.emit('user-connected', socket.id);
    
    // Пересылаем offer между клиентами
    socket.on('offer', (offer, targetId) => {
        console.log(`Offer от ${socket.id} к ${targetId}`);
        socket.to(targetId).emit('offer', offer, socket.id);
    });
    
    // Пересылаем answer между клиентами
    socket.on('answer', (answer, targetId) => {
        console.log(`Answer от ${socket.id} к ${targetId}`);
        socket.to(targetId).emit('answer', answer, socket.id);
    });
    
    // Пересылаем ICE-кандидаты
    socket.on('ice-candidate', (candidate, targetId) => {
        console.log(`ICE candidate от ${socket.id} к ${targetId}`);
        socket.to(targetId).emit('ice-candidate', candidate, socket.id);
    });
    
    // Обработка отключения
    socket.on('disconnect', () => {
        console.log(`Пользователь отключен: ${socket.id}`);
        socket.broadcast.emit('user-disconnected', socket.id);
    });
});

// Запуск сервера
server.listen(PORT, () => {
    console.log(`Сервер запущен на порту ${PORT}`);
    console.log(`Откройте http://localhost:${PORT} в двух браузерах для теста`);
});