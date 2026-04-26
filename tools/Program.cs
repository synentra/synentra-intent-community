using System.Security.Cryptography;
using System.Text;

var inputFile = args[0];
var outputFile = args[1];
var keyString = args[2];

var key = SHA256.HashData(Encoding.UTF8.GetBytes(keyString));

var plaintext = await File.ReadAllBytesAsync(inputFile);

var nonce = RandomNumberGenerator.GetBytes(12);
var ciphertext = new byte[plaintext.Length];
var tag = new byte[16];

using var aes = new AesGcm(key, 16);

aes.Encrypt(nonce, plaintext, ciphertext, tag);

using var fs = new FileStream(outputFile, FileMode.Create);
await fs.WriteAsync(nonce);
await fs.WriteAsync(tag);
await fs.WriteAsync(ciphertext);