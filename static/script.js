// script.js
document.addEventListener('DOMContentLoaded', function() {
    const chatbox = document.getElementById('chatbox');
    const toggleButton = document.getElementById('toggle-button');
    const userInfo = document.getElementById('user-info');
    const userAvatar = document.getElementById('user-avatar');
    const userName = document.getElementById('user-name');

    // Example user data (replace with actual user data)
    const userData = {
        name: 'John Doe',
        avatar: 'user.png'
    };

    // Set user name and avatar
    userName.textContent = userData.name;
    userAvatar.src = `{{ url_for('static', filename='${userData.avatar}') }}`;

    let isOpen = false;

    toggleButton.addEventListener('click', function() {
        isOpen = !isOpen;
        if (isOpen) {
            toggleButton.src = "{{ url_for('static', filename='openclosed.png') }}";
            chatbox.style.display = 'block';
            userInfo.style.display = 'block';
        } else {
            toggleButton.src = "{{ url_for('static', filename='openclosed.png') }}";
            chatbox.style.display = 'none';
            userInfo.style.display = 'none';
        }
    });
});
