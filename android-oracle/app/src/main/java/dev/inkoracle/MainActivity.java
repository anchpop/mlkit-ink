package dev.inkoracle;

import android.app.Activity;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.system.Os;
import android.util.JsonReader;
import android.util.JsonToken;
import android.util.Log;
import android.view.WindowManager;

import com.google.android.gms.tasks.Tasks;
import com.google.mlkit.common.model.DownloadConditions;
import com.google.mlkit.common.model.RemoteModelManager;
import com.google.mlkit.vision.digitalink.common.RecognitionCandidate;
import com.google.mlkit.vision.digitalink.common.RecognitionResult;
import com.google.mlkit.vision.digitalink.recognition.DigitalInkRecognition;
import com.google.mlkit.vision.digitalink.recognition.DigitalInkRecognitionModel;
import com.google.mlkit.vision.digitalink.recognition.DigitalInkRecognitionModelIdentifier;
import com.google.mlkit.vision.digitalink.recognition.DigitalInkRecognizer;
import com.google.mlkit.vision.digitalink.recognition.DigitalInkRecognizerOptions;
import com.google.mlkit.vision.digitalink.recognition.Ink;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.List;

/** File-driven reference oracle: all recognition is performed by ML Kit. */
public final class MainActivity extends Activity {
    private static final String TAG = "InkOracle";
    private static final File INPUT = new File("/sdcard/ink_input.json");
    private static final File OUTPUT = new File("/sdcard/ink_output.json");
    private OracleJob job;

    @Override
    public void onCreate(Bundle state) {
        super.onCreate(state);
        // A CLI launch while locked otherwise becomes TOP_SLEEPING, and Android
        // blocks the SDK downloader as background traffic. Keep just this Activity
        // visible during recognition without dismissing the device's keyguard.
        setShowWhenLocked(true);
        setTurnScreenOn(true);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        // Retain the worker across any configuration change not covered by the manifest.
        job = (OracleJob) getLastNonConfigurationInstance();
        boolean start = job == null;
        if (start) job = new OracleJob();
        job.activity = this;
        if (job.completed) finish();
        else if (start) new Thread(job, "InkOracle-worker").start();
    }

    @Override
    public Object onRetainNonConfigurationInstance() {
        return job;
    }

    @Override
    protected void onDestroy() {
        if (job != null && job.activity == this) job.activity = null;
        // Do not close the recognizer or interrupt a download on Activity recreation.
        super.onDestroy();
    }

    private static final class OracleJob implements Runnable {
        // These fields are accessed only on the main thread.
        private MainActivity activity;
        private boolean completed;
        private final Handler main = new Handler(Looper.getMainLooper());

        @Override
        public void run() {
            JSONObject output = new JSONObject();
            String stage = "read_input";
            try {
                output.put("status", "error");
                output.put("inputSha256", JSONObject.NULL);
                output.put("language", JSONObject.NULL);
                output.put("modelIdentifier", JSONObject.NULL);
                output.put("candidates", new JSONArray());
                byte[] bytes;
                try (FileInputStream input = new FileInputStream(INPUT);
                     ByteArrayOutputStream buffer = new ByteArrayOutputStream()) {
                    byte[] block = new byte[8192];
                    int count;
                    while ((count = input.read(block)) != -1) buffer.write(block, 0, count);
                    bytes = buffer.toByteArray();
                }
                output.put("inputSha256", sha256(bytes));
                stage = "validate_input";
                Ink ink = parseInput(bytes, output);
                String language = output.getString("language");
                stage = "resolve_model";
                DigitalInkRecognitionModelIdentifier identifier =
                        DigitalInkRecognitionModelIdentifier.fromLanguageTag(language);
                if (identifier == null) {
                    throw new IllegalArgumentException("No ML Kit model for language: " + language);
                }
                output.put("modelIdentifier", identifier.getLanguageTag());
                DigitalInkRecognitionModel model =
                        DigitalInkRecognitionModel.builder(identifier).build();
                RemoteModelManager manager = RemoteModelManager.getInstance();
                stage = "check_model";
                Log.i(TAG, "Checking model " + identifier.getLanguageTag());
                if (!Tasks.await(manager.isModelDownloaded(model))) {
                    stage = "download_model";
                    Log.i(TAG, "Downloading model " + identifier.getLanguageTag());
                    Tasks.await(manager.download(model, new DownloadConditions.Builder().build()));
                }
                stage = "recognize";
                try (DigitalInkRecognizer recognizer = DigitalInkRecognition.getClient(
                        DigitalInkRecognizerOptions.builder(model).build())) {
                    RecognitionResult result = Tasks.await(recognizer.recognize(ink));
                    JSONArray candidates = new JSONArray();
                    for (RecognitionCandidate candidate : result.getCandidates()) {
                        JSONObject item = new JSONObject();
                        item.put("text", candidate.getText());
                        Float score = candidate.getScore();
                        item.put("score", score == null ? JSONObject.NULL : score);
                        logCandidate(candidates.length(), item.toString());
                        candidates.put(item);
                    }
                    output.put("candidates", candidates);
                }
                output.put("status", "ok");
            } catch (Exception failure) {
                Log.e(TAG, "Failed at " + stage, failure);
                if (failure instanceof InterruptedException) Thread.currentThread().interrupt();
                try {
                    Throwable cause = failure;
                    while (cause.getCause() != null && cause.getCause() != cause) {
                        cause = cause.getCause();
                    }
                    output.put("status", "error");
                    output.put("error", new JSONObject()
                            .put("stage", stage)
                            .put("type", cause.getClass().getName())
                            .put("message", cause.getMessage() == null
                                    ? cause.toString() : cause.getMessage()));
                } catch (Exception encodingFailure) {
                    Log.e(TAG, "Could not encode error", encodingFailure);
                }
            } finally {
                try {
                    writeAtomic(output);
                    Log.i(TAG, "Completed status=" + output.optString("status")
                            + " candidates=" + output.optJSONArray("candidates").length()
                            + " output=" + OUTPUT);
                } catch (Exception writeFailure) {
                    // No alternate location can satisfy the runner's contract. Be explicit.
                    Log.e(TAG, "Cannot publish " + OUTPUT
                            + "; verify MANAGE_EXTERNAL_STORAGE appops is allowed", writeFailure);
                } finally {
                    main.post(() -> {
                        completed = true;
                        if (activity != null) activity.finish();
                    });
                }
            }
        }
    }

    private static Ink parseInput(byte[] bytes, JSONObject output) throws Exception {
        try (JsonReader reader = new JsonReader(new InputStreamReader(
                new ByteArrayInputStream(bytes), StandardCharsets.UTF_8))) {
            reader.setLenient(false);
            reader.beginObject();
            boolean hasLanguage = false;
            Ink ink = null;
            while (reader.hasNext()) {
                String field = reader.nextName();
                if (field.equals("language")) {
                    require(!hasLanguage, "Duplicate language field");
                    require(reader.peek() == JsonToken.STRING, "language must be a string");
                    String language = reader.nextString();
                    output.put("language", language);
                    require(!language.trim().isEmpty(), "language must not be empty");
                    hasLanguage = true;
                } else if (field.equals("strokes")) {
                    require(ink == null, "Duplicate strokes field");
                    Ink.Builder builder = Ink.builder();
                    reader.beginArray();
                    int count = 0;
                    while (reader.hasNext()) builder.addStroke(readStroke(reader, count++));
                    reader.endArray();
                    require(count > 0, "strokes must contain at least one stroke");
                    ink = builder.build();
                } else {
                    reader.skipValue();
                }
            }
            reader.endObject();
            require(reader.peek() == JsonToken.END_DOCUMENT, "Unexpected trailing JSON content");
            require(hasLanguage, "Missing language");
            require(ink != null, "Missing strokes");
            return ink;
        }
    }

    private static Ink.Stroke readStroke(JsonReader reader, int index) throws IOException {
        List<Float> x = null;
        List<Float> y = null;
        List<Long> t = null;
        String where = "strokes[" + index + "]";
        reader.beginObject();
        while (reader.hasNext()) {
            String field = reader.nextName();
            switch (field) {
                case "x":
                    require(x == null, where + ": duplicate x");
                    x = readCoordinates(reader, where + ".x");
                    break;
                case "y":
                    require(y == null, where + ": duplicate y");
                    y = readCoordinates(reader, where + ".y");
                    break;
                case "t":
                    require(t == null, where + ": duplicate t");
                    t = new ArrayList<>();
                    reader.beginArray();
                    while (reader.hasNext()) {
                        String position = where + ".t[" + t.size() + "]";
                        require(reader.peek() == JsonToken.NUMBER,
                                position + " must be a signed 64-bit integer");
                        String value = reader.nextString();
                        try {
                            t.add(Long.parseLong(value));
                        } catch (NumberFormatException invalid) {
                            throw new IllegalArgumentException(
                                    position + " must be a signed 64-bit integer: " + value, invalid);
                        }
                    }
                    reader.endArray();
                    break;
                default:
                    reader.skipValue();
            }
        }
        reader.endObject();
        require(x != null && y != null, where + " requires x and y arrays");
        require(!x.isEmpty() && x.size() == y.size(),
                where + ": x and y must have the same nonzero length");
        require(t == null || t.size() == x.size(), where + ": t must have the same length as x/y");
        Ink.Stroke.Builder stroke = Ink.Stroke.builder();
        for (int i = 0; i < x.size(); i++) {
            // Absence of t is meaningful to the SDK; never synthesize timestamps.
            stroke.addPoint(t == null ? Ink.Point.create(x.get(i), y.get(i))
                    : Ink.Point.create(x.get(i), y.get(i), t.get(i)));
        }
        return stroke.build();
    }

    private static List<Float> readCoordinates(JsonReader reader, String where) throws IOException {
        List<Float> coordinates = new ArrayList<>();
        reader.beginArray();
        while (reader.hasNext()) {
            String position = where + "[" + coordinates.size() + "]";
            require(reader.peek() == JsonToken.NUMBER, position + " must be a finite number");
            float value = Float.parseFloat(reader.nextString());
            require(!Float.isNaN(value) && !Float.isInfinite(value),
                    position + " must be finite and representable as an SDK float");
            coordinates.add(value);
        }
        reader.endArray();
        return coordinates;
    }

    private static void require(boolean condition, String message) {
        if (!condition) throw new IllegalArgumentException(message);
    }

    private static String sha256(byte[] bytes) throws Exception {
        byte[] digest = MessageDigest.getInstance("SHA-256").digest(bytes);
        StringBuilder hex = new StringBuilder(64);
        for (byte value : digest) {
            hex.append(Character.forDigit((value & 0xff) >>> 4, 16));
            hex.append(Character.forDigit(value & 0xf, 16));
        }
        return hex.toString();
    }

    private static void writeAtomic(JSONObject output) throws Exception {
        File temporary = File.createTempFile(".ink_output.", ".tmp", OUTPUT.getParentFile());
        try {
            try (FileOutputStream stream = new FileOutputStream(temporary)) {
                stream.write((output.toString() + "\n").getBytes(StandardCharsets.UTF_8));
                stream.getFD().sync();
            }
            // Same-directory POSIX rename atomically replaces any previous complete result.
            Os.rename(temporary.getAbsolutePath(), OUTPUT.getAbsolutePath());
        } finally {
            if (temporary.exists() && !temporary.delete()) {
                Log.w(TAG, "Could not remove temporary output " + temporary);
            }
        }
    }

    private static void logCandidate(int index, String json) {
        // Stay below logcat's per-entry byte limit, including for non-ASCII candidates.
        for (int start = 0, part = 0; start < json.length(); part++) {
            int end = Math.min(start + 900, json.length());
            if (end < json.length() && Character.isHighSurrogate(json.charAt(end - 1))) end--;
            Log.i(TAG, "candidate[" + index + "] part[" + part + "] " + json.substring(start, end));
            start = end;
        }
    }
}
