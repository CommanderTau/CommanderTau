const localVideo = document.getElementById('localVideo');
const remoteVideo = document.getElementById('remoteVideo');
const startButton = document.getElementById('startButton');
const callButton = document.getElementById('callButton');
const hangupButton = document.getElementById('hangupButton');
const usersList = document.getElementById('usersList');

let localStream;
let remoteStream;
let peerConnection;
let currentUserId;

const configuration = {
    iceServers: [
        { urls: 'stun:stun.l.google.com:19302' },
        { urls: 'stun:stun1.l.google.com:19302' },
        { urls: 'stun:stun2.l.google.com:19302' }
    ]
};

const socket = io();
startButton.addEventListener('click', start);
callButton.addEventListener('click', call);
hangupButton.addEventListener('click', hangup);

async function start() {
    try {
        localStream = await navigator.mediaDevices.getUserMedia({
            video: true,
            audio: true
        });
        localVideo.srcObject = localStream;
        startButton.disabled = true;
        callButton.disabled = false;
    } catch (err) {
        console.error('Ошибка доступа к медиаустройствам:', err);
    }
}

async function call() {
    callButton.disabled = true;
    hangupButton.disabled = false;
    peerConnection = new RTCPeerConnection(configuration);
    localStream.getTracks().forEach(track => {
        peerConnection.addTrack(track, localStream);
    });
    peerConnection.ontrack = event => {
        remoteStream = new MediaStream();
        event.streams[0].getTracks().forEach(track => {
            remoteStream.addTrack(track);
        });
        remoteVideo.srcObject = remoteStream;
    };
    peerConnection.onicecandidate = event => {
        if (event.candidate) {
            const targetId = document.querySelector('#usersList option:checked').value;
            socket.emit('ice-candidate', {
                target: targetId,
                candidate: event.candidate
            });
        }
    };

    try {
        const offer = await peerConnection.createOffer();
        await peerConnection.setLocalDescription(offer);
        
        // Отправляем offer выбранному пользователю
        const targetId = document.querySelector('#usersList option:checked').value;
        socket.emit('offer', {
            target: targetId,
            offer: peerConnection.localDescription
        });
    } catch (err) {
        console.error('Ошибка создания offer:', err);
    }
}

function hangup() {
    if (peerConnection) {
        peerConnection.close();
        peerConnection = null;
    }
    if (remoteVideo.srcObject) {
        remoteVideo.srcObject.getTracks().forEach(track => track.stop());
        remoteVideo.srcObject = null;
    }
    hangupButton.disabled = true;
    callButton.disabled = false;
}

socket.on('your-id', (id) => {
    currentUserId = id;
    console.log('Ваш ID:', id);
});
socket.on('user-connected', (userId) => {
    console.log('Подключился новый пользователь:', userId);
    const option = document.createElement('option');
    option.value = userId;
    option.textContent = userId;
    usersList.appendChild(option);
});
socket.on('user-disconnected', (userId) => {
    console.log('Пользователь отключился:', userId);
    const option = usersList.querySelector(`option[value="${userId}"]`);
    if (option) option.remove();
    if (peerConnection && peerConnection.connectionState === 'connected') {
        hangup();
    }
});
socket.on('offer', async (data) => {
    if (!peerConnection) {
        peerConnection = new RTCPeerConnection(configuration);
        localStream.getTracks().forEach(track => {
            peerConnection.addTrack(track, localStream);
        });
        peerConnection.ontrack = event => {
            remoteStream = new MediaStream();
            event.streams[0].getTracks().forEach(track => {
                remoteStream.addTrack(track);
            });
            remoteVideo.srcObject = remoteStream;
        };
        peerConnection.onicecandidate = event => {
            if (event.candidate) {
                socket.emit('ice-candidate', {
                    target: data.sender,
                    candidate: event.candidate
                });
            }
        };
    }

    await peerConnection.setRemoteDescription(new RTCSessionDescription(data.offer));
    const answer = await peerConnection.createAnswer();
    await peerConnection.setLocalDescription(answer);
    socket.emit('answer', {
        target: data.sender,
        answer: peerConnection.localDescription
    });
    callButton.disabled = true;
    hangupButton.disabled = false;
});
socket.on('answer', async (data) => {
    await peerConnection.setRemoteDescription(new RTCSessionDescription(data.answer));
});
socket.on('ice-candidate', async (data) => {
    try {
        if (peerConnection) {
            await peerConnection.addIceCandidate(new RTCIceCandidate(data.candidate));
        }
    } catch (err) {
        console.error('Ошибка добавления ICE кандидата:', err);
    }
});