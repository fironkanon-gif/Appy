package org.gothicocr;

import org.tensorflow.lite.Interpreter;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public final class TFLiteBridge implements AutoCloseable {
    private Interpreter interpreter;
    private final int outputBytes;

    public TFLiteBridge(String modelPath, int outputBytes) throws IOException {
        if (outputBytes <= 0) throw new IllegalArgumentException("outputBytes must be > 0");
        this.outputBytes = outputBytes;
        MappedByteBuffer model = loadModel(modelPath);
        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(Math.max(1, Math.min(4, Runtime.getRuntime().availableProcessors())));
        interpreter = new Interpreter(model, options);
        interpreter.allocateTensors();
    }

    private static MappedByteBuffer loadModel(String path) throws IOException {
        File file = new File(path);
        if (!file.isFile()) throw new IOException("Model file not found: " + path);
        try (FileInputStream input = new FileInputStream(file);
             FileChannel channel = input.getChannel()) {
            return channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size());
        }
    }

    public byte[] run(byte[] inputBytes) {
        if (interpreter == null) throw new IllegalStateException("Interpreter is closed");
        ByteBuffer input = ByteBuffer.allocateDirect(inputBytes.length).order(ByteOrder.nativeOrder());
        input.put(inputBytes);
        input.rewind();

        ByteBuffer output = ByteBuffer.allocateDirect(outputBytes).order(ByteOrder.nativeOrder());
        interpreter.run(input, output);
        output.rewind();

        byte[] result = new byte[outputBytes];
        output.get(result);
        return result;
    }

    @Override
    public void close() {
        if (interpreter != null) {
            interpreter.close();
            interpreter = null;
        }
    }
}
