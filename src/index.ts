import { Elysia, StatusMap } from "elysia";
import { bearer } from "@elysiajs/bearer";
import * as tf from "@tensorflow/tfjs";
import * as nsfwjs from "nsfwjs";
import sharp from "sharp";
import cors from "@elysiajs/cors";
import swagger from "@elysiajs/swagger";

const modelName = Bun.env.MODEL_NAME || "InceptionV3";

let model: nsfwjs.NSFWJS;

if (modelName === "InceptionV3") {
  model = await nsfwjs.load(modelName, { type: "graph" });
} else {
  model = await nsfwjs.load(modelName);
}

const app = new Elysia()
  .use(bearer())
  .use(cors())
  .use(swagger())
  .post(
    "/classify",
    async (ctx) => {
      const imageBuffer = await ctx.request.arrayBuffer();
      const image = sharp(imageBuffer);
      const imageMetadata = await image.metadata();
      const numChannels = 3;
      if (!imageMetadata.width || !imageMetadata.height) {
        throw new Error("Image width is not available");
      }
      const numPixels = imageMetadata.width * imageMetadata.height;
      const values = new Int32Array(numPixels * numChannels);
      const rawValues = await image.raw().toBuffer();
      for (let i = 0; i < numPixels; i++) {
        for (let c = 0; c < numChannels; c++) {
          values[i * numChannels + c] = rawValues[i * 4 + c];
        }
      }
      const imageTensor = tf.tensor3d(
        values,
        [imageMetadata.height, imageMetadata.width, numChannels],
        "int32"
      );

      const predictions = await model.classify(imageTensor);
      let result: Record<string, number> = {};
      predictions.forEach((p) => {
        result[p.className] = p.probability;
      });
      return result;
    },
    {
      beforeHandle(context) {
        console.log(
          `${context.request.method} ${
            context.request.url
          } ${context.request.headers.get(
            "User-Agent"
          )} ${context.request.headers.get("Content-Type")}`
        );
        if (!context.bearer || context.bearer !== Bun.env.ACCESS_TOKEN) {
          context.set.status = StatusMap.Unauthorized;
          context.set.headers[
            "WWW-Authenticate"
          ] = `Bearer realm='sign', error="invalid_request"`;
          return "Unauthorized";
        }
      },
    }
  );

app.listen({
  hostname: Bun.env.HOSTNAME || "localhost",
  port: Bun.env.PORT || 3000,
});

console.log(
  `🦊 Elysia is running at ${app.server?.hostname}:${app.server?.port}`
);
