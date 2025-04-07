#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <pthread.h>
#include <termios.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <linux/videodev2.h>
#include <time.h>

#define DEVICE "/dev/video0"
#define WIDTH 640
#define HEIGHT 480
#define FRAME_COUNT 100
#define PHOTOS_DIR "photos"

typedef struct {
    int fd;
    void *buffer;
    size_t buffer_length;
    int stop;
    int frame_number;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    int frame_ready;
    struct v4l2_buffer buf_info;
} SharedData;

void *capture_thread(void *arg) {
    SharedData *data = (SharedData *)arg;

    while (!data->stop) {
        pthread_mutex_lock(&data->mutex);
        if (ioctl(data->fd, VIDIOC_QBUF, &data->buf_info) == -1) {
            perror("Failed to queue buffer");
            data->stop = 1;
        }

        if (ioctl(data->fd, VIDIOC_DQBUF, &data->buf_info) == -1) {
            perror("Failed to dequeue buffer");
            data->stop = 1;
        }

        data->frame_ready = 1;
        pthread_cond_signal(&data->cond);
        pthread_mutex_unlock(&data->mutex);

        usleep(10000); // Optional delay
    }

    return NULL;
}

void *write_thread(void *arg) {
    SharedData *data = (SharedData *)arg;

    while (1) {
        pthread_mutex_lock(&data->mutex);
        while (!data->frame_ready && !data->stop) {
            pthread_cond_wait(&data->cond, &data->mutex);
        }

        if (data->stop) {
            pthread_mutex_unlock(&data->mutex);
            break;
        }

        char filename[256];
        snprintf(filename, sizeof(filename), "%s/frame_%04d.jpg", PHOTOS_DIR, data->frame_number++);

        FILE *fp = fopen(filename, "wb");
        if (fp) {
            fwrite(data->buffer, data->buf_info.bytesused, 1, fp);
            fclose(fp);
            printf("Saved: %s\n", filename);
        } else {
            perror("Failed to write frame");
        }

        data->frame_ready = 0;
        pthread_mutex_unlock(&data->mutex);
    }

    data->stop = 1;  // Signal other threads to stop when done
    pthread_cond_broadcast(&data->cond);
    return NULL;
}

void *input_listener_thread(void *arg) {
    SharedData *data = (SharedData *)arg;

    struct termios oldt, newt;
    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);

    printf("Press 'q' to stop...\n");

    while (!data->stop) {
        char c = getchar();
        if (c == 'q' || c == 'Q') {
            pthread_mutex_lock(&data->mutex);
            data->stop = 1;
            pthread_cond_broadcast(&data->cond);
            pthread_mutex_unlock(&data->mutex);
            break;
        }
    }

    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    return NULL;
}

int main() {
    // Create photos directory
    if (mkdir(PHOTOS_DIR, 0755) == -1 && errno != EEXIST) {
        perror("Failed to create photos directory");
        return 1;
    }

    int fd = open(DEVICE, O_RDWR);
    if (fd == -1) {
        perror("Failed to open video device");
        return 1;
    }

    struct v4l2_capability cap;
    if (ioctl(fd, VIDIOC_QUERYCAP, &cap) == -1) {
        perror("Failed to query capabilities");
        close(fd);
        return 1;
    }

    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
        fprintf(stderr, "Device does not support video capture\n");
        close(fd);
        return 1;
    }

    struct v4l2_format fmt = {0};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = WIDTH;
    fmt.fmt.pix.height = HEIGHT;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;

    if (ioctl(fd, VIDIOC_S_FMT, &fmt) == -1) {
        perror("Failed to set format");
        close(fd);
        return 1;
    }

    struct v4l2_requestbuffers req = {0};
    req.count = 1;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;

    if (ioctl(fd, VIDIOC_REQBUFS, &req) == -1) {
        perror("Failed to request buffer");
        close(fd);
        return 1;
    }

    struct v4l2_buffer buf = {0};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = 0;

    if (ioctl(fd, VIDIOC_QUERYBUF, &buf) == -1) {
        perror("Failed to query buffer");
        close(fd);
        return 1;
    }

    void *buffer = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, buf.m.offset);
    if (buffer == MAP_FAILED) {
        perror("Failed to map buffer");
        close(fd);
        return 1;
    }

    if (ioctl(fd, VIDIOC_STREAMON, &buf.type) == -1) {
        perror("Failed to start stream");
        munmap(buffer, buf.length);
        close(fd);
        return 1;
    }

    SharedData data = {
        .fd = fd,
        .buffer = buffer,
        .buffer_length = buf.length,
        .stop = 0,
        .frame_number = 0,
        .frame_ready = 0,
        .buf_info = buf
    };
    pthread_mutex_init(&data.mutex, NULL);
    pthread_cond_init(&data.cond, NULL);

    pthread_t capture, writer, listener;
    pthread_create(&capture, NULL, capture_thread, &data);
    pthread_create(&writer, NULL, write_thread, &data);
    pthread_create(&listener, NULL, input_listener_thread, &data);

    pthread_join(capture, NULL);
    pthread_join(writer, NULL);
    pthread_join(listener, NULL);

    ioctl(fd, VIDIOC_STREAMOFF, &buf.type);
    munmap(buffer, buf.length);
    close(fd);
    pthread_mutex_destroy(&data.mutex);
    pthread_cond_destroy(&data.cond);

    printf("Done capturing %d frames.\n", data.frame_number);
    return 0;
}
