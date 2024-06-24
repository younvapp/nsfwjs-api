import { Elysia, StatusMap } from "elysia";
import { bearer } from "@elysiajs/bearer";
import * as tf from "@tensorflow/tfjs-node";
import * as nsfwjs from "nsfwjs";
import sharp from "sharp";
import cors from "@elysiajs/cors";
import swagger from "@elysiajs/swagger";
import * as jpeg from "jpeg-js";

await tf.enableProdMode();
await tf.ready();

const model = await nsfwjs.load(
  Bun.pathToFileURL(Bun.env.MODEL || "./models/nsfwjs/").toString(),
  { size: 299 }
);
const app = new Elysia()
  .use(bearer())
  .use(cors())
  .use(swagger())
  .post(
    "/classify",
    async (ctx) => {
      const imageBuffer = await ctx.request.arrayBuffer();
      const image = await sharp(imageBuffer).raw().jpeg().toBuffer();
      const decoded = jpeg.decode(image);
      const { width, height, data } = decoded;
      const buffer = new Uint8Array(width * height * 3);
      let offset = 0;
      for (let i = 0; i < buffer.length; i += 3) {
        buffer[i] = data[offset];
        buffer[i + 1] = data[offset + 1];
        buffer[i + 2] = data[offset + 2];
        offset += 4;
      }
      let imageTensor = tf.tensor3d(buffer, [height, width, 3]);

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
