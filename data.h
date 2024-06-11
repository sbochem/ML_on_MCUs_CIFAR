#ifndef CIFAR_IMAGE_H
#define CIFAR_IMAGE_H

static const unsigned int cifar_image_len = 784;

const unsigned char cifar_image[] = {
			    // Channel 1
			   90, 90, 91, 91, 93, 94, 94, 95, 99, 102, 103, 105, 106, 108, 108, 107, 107, 106, 104, 112, 117, 111, 108, 112, 111, 110, 110, 107, 106, 102, 99, 99,
			   92, 93, 94, 94, 94, 96, 96, 96, 101, 104, 104, 105, 104, 105, 106, 106, 107, 107, 111, 116, 117, 111, 107, 110, 115, 121, 134, 150, 146, 113, 102, 101,
			   93, 93, 94, 93, 94, 95, 95, 96, 99, 102, 102, 102, 100, 101, 101, 103, 104, 106, 111, 112, 112, 109, 106, 108, 117, 133, 166, 187, 155, 113, 102, 100,
			   95, 93, 94, 94, 95, 95, 95, 96, 98, 100, 99, 99, 99, 99, 99, 101, 102, 104, 109, 108, 108, 104, 107, 113, 129, 161, 184, 169, 126, 109, 102, 101,
			   96, 94, 94, 95, 96, 96, 95, 96, 96, 96, 96, 96, 96, 96, 96, 97, 98, 99, 103, 103, 104, 88, 96, 128, 165, 186, 180, 135, 109, 105, 102, 103,
			   96, 95, 95, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 98, 98, 103, 104, 100, 149, 189, 188, 154, 108, 101, 102, 103, 105,
			   96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 97, 97, 96, 96, 97, 106, 129, 114, 147, 188, 163, 116, 99, 101, 110, 111, 105,
			   96, 95, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 98, 99, 99, 96, 102, 123, 133, 122, 133, 167, 125, 102, 104, 110, 149, 145, 105,
			   96, 95, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 95, 96, 97, 99, 102, 105, 136, 167, 137, 126, 115, 129, 113, 105, 117, 158, 181, 141, 101,
			   96, 95, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 95, 96, 99, 103, 111, 140, 183, 189, 140, 128, 104, 108, 111, 118, 163, 194, 166, 113, 99,
			   96, 96, 96, 96, 97, 97, 97, 97, 96, 96, 96, 96, 96, 96, 99, 107, 125, 153, 188, 203, 185, 138, 131, 105, 103, 122, 149, 184, 180, 134, 103, 99,
			   97, 96, 96, 96, 97, 97, 97, 97, 97, 97, 97, 97, 97, 100, 106, 128, 154, 178, 204, 205, 187, 157, 148, 123, 111, 139, 161, 176, 153, 113, 103, 103,
			   97, 96, 96, 96, 97, 97, 98, 98, 98, 98, 98, 99, 99, 110, 133, 170, 159, 128, 190, 206, 192, 175, 164, 162, 151, 140, 146, 159, 133, 108, 105, 107,
			   100, 98, 98, 96, 99, 111, 115, 122, 120, 120, 128, 129, 134, 155, 175, 197, 166, 108, 161, 204, 197, 185, 188, 199, 197, 169, 119, 92, 119, 112, 106, 106,
			   98, 97, 98, 97, 108, 125, 148, 175, 177, 182, 189, 189, 190, 195, 189, 183, 163, 113, 129, 191, 204, 196, 197, 200, 202, 189, 145, 105, 120, 114, 107, 102,
			   93, 96, 98, 115, 106, 101, 158, 193, 198, 205, 206, 207, 209, 213, 206, 199, 168, 114, 118, 171, 197, 194, 194, 192, 191, 191, 189, 143, 114, 112, 107, 102,
			   94, 102, 144, 175, 159, 168, 175, 174, 180, 190, 199, 202, 207, 213, 210, 212, 177, 129, 123, 148, 175, 174, 174, 158, 146, 140, 132, 105, 109, 112, 107, 103,
			   103, 146, 204, 189, 165, 175, 170, 162, 179, 195, 209, 210, 207, 202, 201, 208, 180, 142, 124, 122, 157, 150, 123, 83, 76, 109, 114, 112, 110, 108, 106, 104,
			   112, 167, 187, 139, 110, 148, 167, 172, 181, 193, 208, 209, 207, 203, 204, 209, 181, 142, 115, 109, 135, 96, 70, 43, 24, 97, 113, 107, 106, 105, 104, 103,
			   106, 125, 124, 103, 97, 116, 124, 135, 142, 157, 182, 196, 205, 202, 204, 200, 179, 141, 105, 112, 119, 69, 56, 36, 41, 101, 105, 101, 103, 102, 101, 101,
			   100, 102, 104, 107, 106, 102, 102, 106, 107, 112, 130, 173, 204, 201, 199, 190, 179, 115, 88, 136, 166, 153, 142, 99, 96, 113, 104, 100, 101, 99, 98, 98,
			   102, 101, 102, 102, 101, 101, 102, 104, 102, 100, 106, 161, 203, 202, 197, 190, 175, 105, 88, 160, 201, 195, 181, 131, 112, 111, 106, 101, 100, 98, 96, 96,
			   103, 101, 101, 101, 101, 101, 103, 103, 101, 100, 105, 161, 206, 207, 198, 187, 158, 113, 109, 169, 202, 192, 157, 112, 107, 107, 104, 101, 100, 97, 96, 96,
			   103, 101, 101, 101, 101, 101, 103, 103, 101, 100, 104, 161, 209, 210, 196, 180, 136, 106, 110, 164, 199, 182, 133, 108, 105, 102, 101, 101, 100, 97, 96, 96,
			   102, 101, 101, 101, 101, 101, 103, 103, 101, 100, 104, 161, 206, 204, 194, 166, 115, 103, 106, 156, 195, 157, 112, 106, 105, 100, 99, 100, 100, 98, 96, 96,
			   102, 101, 101, 101, 101, 101, 102, 103, 101, 99, 104, 156, 199, 198, 188, 142, 105, 104, 104, 138, 154, 117, 106, 108, 104, 100, 97, 99, 99, 97, 96, 95,
			   102, 101, 101, 101, 101, 101, 102, 103, 101, 99, 104, 153, 192, 194, 173, 118, 105, 105, 105, 113, 113, 104, 105, 108, 105, 101, 97, 98, 98, 96, 94, 94,
			   102, 101, 101, 101, 102, 102, 103, 103, 101, 100, 105, 155, 192, 192, 153, 109, 106, 105, 104, 106, 108, 102, 103, 107, 105, 102, 98, 97, 97, 95, 93, 93,
			   101, 101, 101, 101, 103, 103, 103, 103, 102, 104, 112, 161, 192, 180, 130, 110, 105, 102, 103, 104, 104, 103, 104, 105, 104, 102, 100, 98, 96, 95, 93, 92,
			   100, 100, 101, 101, 103, 103, 103, 103, 104, 111, 117, 140, 155, 147, 117, 110, 106, 101, 103, 103, 103, 103, 104, 104, 102, 101, 99, 97, 96, 94, 92, 91,
			   99, 100, 101, 101, 103, 103, 103, 103, 107, 110, 108, 112, 115, 117, 112, 108, 105, 101, 103, 103, 103, 103, 103, 102, 101, 100, 98, 96, 95, 93, 90, 90,
			   98, 99, 100, 101, 102, 102, 103, 103, 106, 108, 107, 108, 108, 108, 108, 106, 104, 102, 103, 103, 103, 103, 103, 102, 100, 98, 96, 95, 93, 92, 90, 89,
			    // Channel 2
			   123, 123, 124, 124, 125, 128, 128, 129, 130, 129, 130, 132, 133, 136, 136, 135, 135, 135, 136, 135, 135, 136, 137, 137, 135, 133, 133, 132, 134, 131, 128, 129,
			   126, 126, 128, 128, 128, 130, 130, 131, 133, 134, 134, 135, 134, 135, 136, 137, 137, 138, 138, 138, 139, 140, 139, 137, 138, 140, 150, 165, 165, 137, 130, 132,
			   128, 126, 128, 126, 126, 129, 129, 130, 132, 134, 134, 134, 132, 133, 133, 135, 137, 136, 135, 135, 137, 139, 137, 135, 138, 146, 175, 195, 167, 133, 129, 132,
			   129, 126, 128, 128, 129, 129, 129, 130, 133, 134, 134, 134, 133, 133, 133, 135, 136, 136, 135, 135, 136, 132, 132, 135, 145, 171, 191, 178, 141, 130, 130, 132,
			   131, 128, 128, 129, 130, 130, 129, 130, 132, 133, 133, 133, 133, 133, 133, 133, 134, 134, 134, 135, 133, 108, 111, 142, 177, 194, 187, 150, 132, 131, 131, 134,
			   132, 129, 129, 130, 131, 131, 130, 130, 132, 132, 132, 132, 132, 132, 131, 132, 132, 132, 134, 134, 133, 119, 108, 157, 197, 196, 164, 132, 132, 132, 132, 134,
			   132, 129, 130, 130, 131, 131, 130, 130, 131, 131, 131, 131, 131, 132, 133, 132, 131, 131, 133, 133, 133, 142, 119, 152, 197, 178, 137, 131, 132, 134, 134, 132,
			   131, 129, 130, 130, 130, 130, 130, 130, 130, 130, 130, 130, 130, 133, 134, 133, 132, 131, 130, 132, 143, 145, 129, 138, 180, 147, 130, 134, 134, 164, 162, 131,
			   131, 129, 130, 130, 130, 130, 130, 130, 130, 130, 130, 130, 130, 133, 134, 133, 132, 131, 130, 155, 179, 141, 131, 123, 142, 135, 131, 137, 170, 193, 160, 129,
			   131, 129, 130, 130, 130, 130, 130, 130, 130, 130, 130, 130, 130, 133, 133, 133, 133, 136, 155, 192, 192, 138, 128, 112, 120, 130, 140, 172, 198, 178, 135, 129,
			   132, 130, 130, 130, 132, 132, 132, 132, 131, 130, 130, 130, 130, 132, 133, 136, 147, 169, 193, 206, 184, 134, 130, 110, 110, 133, 162, 186, 182, 148, 128, 131,
			   132, 130, 130, 131, 132, 132, 132, 132, 132, 132, 132, 132, 132, 132, 133, 147, 165, 184, 205, 205, 186, 153, 144, 121, 111, 140, 163, 177, 160, 135, 132, 133,
			   132, 130, 130, 131, 132, 131, 132, 132, 132, 132, 132, 132, 133, 138, 152, 182, 162, 126, 191, 207, 192, 174, 159, 155, 145, 133, 140, 161, 148, 136, 135, 135,
			   133, 131, 132, 132, 135, 139, 138, 144, 144, 145, 150, 149, 153, 167, 182, 201, 165, 106, 162, 205, 196, 183, 184, 192, 190, 159, 108, 94, 137, 138, 135, 135,
			   135, 133, 133, 132, 137, 142, 154, 182, 188, 192, 197, 195, 194, 195, 187, 181, 161, 112, 131, 192, 203, 192, 193, 197, 196, 180, 135, 108, 136, 136, 135, 135,
			   138, 135, 132, 141, 116, 102, 155, 194, 201, 207, 209, 211, 212, 211, 202, 196, 166, 113, 121, 172, 196, 189, 190, 190, 188, 187, 184, 151, 132, 135, 135, 134,
			   137, 133, 164, 183, 149, 157, 167, 171, 178, 186, 197, 203, 209, 211, 206, 208, 174, 128, 125, 149, 174, 169, 169, 155, 146, 142, 137, 119, 131, 137, 135, 134,
			   133, 162, 208, 182, 150, 166, 166, 162, 178, 192, 207, 210, 207, 200, 197, 205, 178, 141, 126, 124, 156, 145, 118, 81, 78, 119, 131, 134, 135, 135, 135, 135,
			   137, 178, 187, 132, 104, 151, 177, 180, 188, 200, 212, 209, 205, 199, 201, 206, 178, 141, 117, 111, 133, 91, 62, 39, 29, 113, 138, 136, 135, 133, 132, 133,
			   138, 146, 137, 110, 107, 137, 149, 153, 159, 174, 192, 196, 199, 197, 201, 196, 176, 140, 107, 114, 117, 64, 50, 33, 48, 123, 138, 134, 134, 132, 130, 131,
			   136, 132, 130, 130, 131, 133, 134, 133, 134, 141, 149, 178, 199, 200, 197, 185, 176, 118, 96, 140, 164, 148, 141, 105, 109, 139, 137, 134, 133, 130, 128, 128,
			   135, 133, 133, 132, 132, 133, 135, 135, 135, 135, 134, 174, 202, 203, 197, 184, 174, 114, 102, 167, 199, 192, 187, 149, 135, 138, 136, 133, 132, 129, 126, 128,
			   135, 133, 133, 133, 133, 133, 135, 135, 135, 136, 135, 175, 204, 204, 195, 184, 164, 130, 129, 178, 203, 192, 168, 135, 132, 135, 134, 133, 132, 129, 126, 128,
			   135, 133, 133, 133, 133, 133, 135, 135, 136, 136, 134, 175, 206, 202, 192, 183, 151, 131, 132, 176, 205, 189, 148, 132, 132, 132, 132, 133, 132, 129, 126, 128,
			   135, 133, 133, 133, 133, 133, 135, 135, 135, 136, 134, 175, 203, 194, 191, 177, 139, 133, 132, 172, 207, 172, 134, 133, 135, 132, 132, 133, 132, 130, 126, 126,
			   134, 133, 133, 133, 133, 133, 135, 135, 135, 136, 134, 170, 196, 190, 189, 159, 133, 135, 132, 159, 172, 141, 134, 136, 135, 133, 133, 132, 131, 129, 125, 125,
			   134, 133, 133, 133, 133, 133, 135, 135, 135, 136, 134, 167, 191, 190, 180, 141, 133, 133, 134, 137, 136, 134, 137, 138, 136, 136, 134, 131, 129, 126, 124, 124,
			   134, 133, 133, 133, 134, 134, 135, 135, 135, 136, 134, 167, 192, 192, 165, 135, 135, 133, 135, 134, 135, 135, 136, 138, 138, 137, 134, 130, 128, 125, 123, 123,
			   134, 133, 133, 133, 135, 135, 135, 135, 135, 136, 136, 172, 195, 188, 148, 135, 137, 136, 135, 134, 135, 135, 136, 137, 136, 135, 132, 129, 126, 124, 122, 122,
			   133, 132, 133, 133, 135, 135, 135, 135, 136, 140, 139, 155, 165, 161, 137, 136, 137, 136, 135, 135, 135, 135, 136, 136, 134, 133, 131, 128, 125, 123, 121, 121,
			   131, 132, 133, 133, 135, 135, 135, 135, 135, 135, 131, 135, 137, 138, 135, 136, 137, 136, 135, 135, 135, 135, 135, 134, 133, 132, 130, 126, 124, 122, 119, 120,
			   130, 130, 132, 133, 134, 134, 135, 135, 134, 132, 132, 137, 137, 136, 135, 135, 136, 135, 135, 135, 135, 135, 135, 134, 132, 130, 128, 125, 122, 121, 119, 119,
			    // Channel 3
			   149, 149, 150, 150, 151, 153, 153, 154, 154, 152, 152, 154, 156, 155, 154, 154, 153, 153, 155, 155, 156, 157, 152, 148, 147, 148, 150, 150, 151, 148, 146, 146,
			   153, 152, 153, 153, 153, 155, 155, 156, 157, 156, 156, 157, 156, 155, 155, 155, 156, 155, 154, 158, 160, 159, 158, 157, 153, 148, 154, 172, 175, 151, 148, 152,
			   153, 152, 152, 152, 152, 153, 154, 155, 155, 155, 155, 155, 153, 153, 153, 155, 157, 154, 150, 155, 159, 154, 155, 159, 152, 149, 169, 194, 173, 145, 147, 152,
			   155, 152, 153, 153, 154, 154, 154, 155, 155, 154, 154, 154, 154, 154, 154, 156, 158, 156, 152, 157, 156, 138, 142, 154, 156, 173, 185, 178, 148, 142, 147, 152,
			   156, 153, 153, 154, 155, 155, 154, 155, 155, 153, 153, 153, 153, 155, 156, 156, 157, 156, 156, 161, 152, 105, 106, 150, 183, 196, 188, 158, 145, 146, 149, 152,
			   157, 154, 154, 155, 156, 156, 155, 155, 154, 152, 152, 152, 152, 154, 155, 155, 156, 156, 161, 164, 150, 110, 93, 153, 197, 201, 174, 149, 152, 150, 150, 151,
			   157, 154, 155, 155, 156, 156, 155, 155, 155, 153, 153, 153, 153, 152, 153, 154, 156, 158, 161, 157, 145, 137, 104, 139, 194, 187, 156, 156, 154, 150, 149, 150,
			   157, 154, 155, 155, 155, 155, 155, 155, 155, 155, 155, 155, 155, 152, 152, 155, 157, 159, 155, 149, 149, 141, 116, 123, 177, 158, 153, 159, 152, 176, 176, 150,
			   157, 154, 155, 155, 155, 155, 155, 155, 155, 155, 155, 155, 155, 155, 156, 156, 156, 154, 147, 164, 179, 132, 116, 111, 141, 147, 152, 153, 180, 203, 175, 150,
			   157, 154, 155, 155, 155, 155, 155, 155, 155, 155, 155, 155, 155, 158, 158, 155, 152, 152, 163, 194, 187, 123, 111, 100, 118, 140, 157, 180, 201, 187, 152, 151,
			   157, 155, 155, 155, 157, 157, 157, 157, 156, 155, 155, 155, 155, 158, 157, 153, 158, 175, 195, 202, 175, 116, 111, 96, 104, 138, 173, 189, 184, 160, 148, 153,
			   158, 155, 155, 156, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 153, 158, 167, 181, 203, 200, 175, 137, 124, 103, 99, 137, 167, 178, 166, 151, 154, 155,
			   157, 154, 155, 155, 156, 155, 155, 155, 156, 156, 156, 156, 156, 160, 168, 186, 155, 115, 186, 200, 182, 160, 139, 132, 128, 123, 136, 163, 160, 157, 159, 156,
			   154, 151, 153, 154, 157, 155, 149, 155, 158, 160, 164, 162, 165, 178, 188, 198, 155, 94, 153, 195, 185, 171, 166, 171, 171, 146, 99, 95, 148, 156, 156, 156,
			   156, 153, 152, 150, 151, 145, 150, 180, 190, 196, 200, 197, 195, 195, 185, 176, 151, 100, 117, 179, 191, 181, 179, 180, 181, 167, 124, 108, 144, 149, 153, 155,
			   165, 157, 151, 154, 115, 96, 145, 187, 197, 204, 206, 209, 212, 210, 200, 192, 156, 100, 107, 158, 184, 177, 176, 175, 176, 177, 178, 153, 141, 148, 153, 155,
			   161, 149, 172, 182, 136, 143, 154, 163, 170, 178, 192, 200, 208, 210, 204, 203, 165, 115, 111, 135, 161, 157, 157, 145, 139, 139, 137, 126, 143, 151, 153, 154,
			   148, 166, 201, 169, 135, 154, 159, 158, 172, 187, 202, 206, 203, 198, 195, 199, 168, 128, 112, 110, 143, 132, 108, 76, 78, 123, 138, 146, 150, 151, 153, 154,
			   147, 179, 177, 118, 99, 152, 182, 185, 190, 201, 210, 204, 198, 196, 198, 200, 169, 129, 104, 97, 119, 79, 55, 39, 35, 123, 153, 153, 153, 152, 151, 152,
			   154, 155, 138, 110, 116, 153, 168, 166, 170, 184, 195, 193, 192, 194, 199, 192, 167, 128, 95, 100, 104, 51, 45, 37, 56, 139, 157, 155, 154, 152, 148, 148,
			   158, 150, 144, 144, 149, 156, 158, 153, 152, 158, 160, 179, 192, 198, 197, 180, 170, 113, 91, 132, 154, 137, 138, 113, 121, 156, 157, 155, 154, 151, 147, 148,
			   157, 153, 153, 152, 153, 155, 156, 156, 157, 158, 151, 179, 197, 203, 197, 180, 173, 120, 109, 167, 192, 184, 189, 161, 148, 154, 154, 154, 153, 150, 148, 149,
			   157, 154, 154, 154, 154, 154, 155, 156, 158, 159, 153, 181, 199, 201, 193, 182, 169, 143, 140, 181, 200, 189, 172, 148, 146, 152, 153, 153, 153, 150, 148, 150,
			   157, 154, 154, 154, 154, 154, 156, 156, 158, 159, 152, 181, 201, 196, 188, 186, 163, 149, 145, 182, 205, 190, 156, 146, 148, 151, 153, 154, 153, 150, 148, 150,
			   156, 154, 154, 154, 154, 154, 156, 156, 158, 159, 152, 181, 198, 186, 188, 185, 158, 155, 147, 181, 212, 180, 146, 148, 152, 152, 154, 154, 153, 151, 148, 149,
			   156, 154, 154, 154, 154, 154, 156, 156, 158, 159, 152, 176, 191, 182, 190, 172, 154, 157, 150, 170, 181, 154, 150, 153, 154, 155, 156, 154, 152, 150, 147, 149,
			   156, 154, 154, 154, 154, 154, 156, 156, 158, 159, 153, 173, 186, 185, 186, 158, 155, 154, 152, 151, 149, 152, 156, 156, 157, 158, 157, 154, 150, 148, 146, 147,
			   156, 154, 154, 154, 155, 155, 156, 156, 158, 158, 150, 172, 187, 190, 174, 153, 156, 154, 154, 151, 152, 155, 157, 157, 159, 159, 157, 153, 151, 149, 146, 148,
			   155, 154, 154, 154, 156, 156, 156, 156, 158, 157, 150, 177, 193, 191, 157, 151, 157, 157, 156, 155, 155, 156, 157, 158, 157, 156, 154, 154, 153, 150, 148, 149,
			   154, 153, 154, 154, 156, 156, 156, 156, 156, 157, 153, 163, 169, 168, 149, 152, 158, 158, 156, 156, 156, 156, 157, 157, 155, 154, 152, 153, 152, 150, 148, 148,
			   153, 153, 154, 154, 156, 156, 156, 156, 153, 149, 145, 147, 149, 150, 148, 153, 157, 158, 156, 156, 156, 156, 156, 155, 154, 153, 151, 151, 151, 149, 146, 147,
			   152, 151, 153, 154, 155, 155, 156, 156, 151, 146, 148, 154, 156, 152, 151, 154, 156, 157, 156, 156, 156, 156, 156, 155, 153, 151, 150, 150, 149, 148, 146, 146,
			};