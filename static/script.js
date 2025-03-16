document.addEventListener('DOMContentLoaded', function() {
    const videoDropdown = document.getElementById('videoDropdown');
    const timestampDropdown = document.getElementById('timestampDropdown');

    videoDropdown.addEventListener('change', function() {
        const selectedVideo = this.value;
        updateTimestamps(selectedVideo);
    });

    function updateTimestamps(video) {
        // Clear existing timestamps
        timestampDropdown.innerHTML = '';

        // Example timestamps for each video
        const timestamps = {
            'video1': ['00:00', '01:30', '02:45'],
            'video2': ['00:15', '01:00', '03:00'],
            'video3': ['00:30', '01:15', '02:30']
        };

        // Populate timestamps based on selected video
        if (timestamps[video]) {
            timestamps[video].forEach(function(timestamp) {
                const option = document.createElement('option');
                option.value = timestamp;
                option.textContent = timestamp;
                timestampDropdown.appendChild(option);
            });
        }
    }
});