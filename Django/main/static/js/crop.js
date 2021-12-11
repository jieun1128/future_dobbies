var clsImage;
		var iCropLeft, iCropTop, iCropWidth, iCropHeight;
		var rect;
		// 로컬 이미지 파일을 Canvas 에 로드한다.
		function LoadImage()
		{
			if( typeof window.FileReader !== 'function' )
			{
				alert("FileReader is not supported");
				return;
			}

			var inputFile = document.getElementById('image_file');
			var clsFileReader = new FileReader();
			clsFileReader.onload = function(){
				clsImage = new Image();
				clsImage.onload = function(){
					var canvas = document.getElementById("canvas");
					canvas.width = clsImage.width;
					canvas.height = clsImage.height;
					rect = canvas.getBoundingClientRect()
						
					iCropLeft = 0;
					iCropTop = 0;
					iCropWidth = clsImage.width;
					iCropHeight = clsImage.height;
					iImageWidth = clsImage.width;
					iImageHeight = clsImage.height;

					DrawCropRect();
					AddCropMoveEvent();
				};

				clsImage.src = clsFileReader.result;
			};

			clsFileReader.readAsDataURL(inputFile.files[0]);
		}

		// 로컬 이미지 파일과 Crop 을 위한 사각형 박스를 그려준다.
		function DrawCropRect()
		{
			var canvas = document.getElementById("canvas");
			var ctx = canvas.getContext("2d");

			ctx.drawImage( clsImage, 0, 0 );

			ctx.strokeStyle = "#ff0000";
			ctx.beginPath();
			ctx.rect(iCropLeft,iCropTop,iCropWidth,iCropHeight);
			ctx.stroke();
			// console.log("Origin", iCropLeft,iCropTop);
			// console.log("rect", iCropWidth, iCropHeight);

		}

		// 이미지를 crop 하여서 하단 Canvas 에 그려준다.
		function CropImage()
		{
			var canvas = document.getElementById("canvas"),
				hidden = document.getElementById('hidden');

			img = new Image();
			img.onload = function(){
				var canvas = document.getElementById("canvas_crop");
				canvas.width = iCropWidth;
				canvas.height = iCropHeight;
				var ctx = canvas.getContext("2d");
				ctx.drawImage( img, iCropLeft, iCropTop, iCropWidth, iCropHeight, 0, 0, iCropWidth, iCropHeight );
				hidden.value = canvas.toDataURL();
			};

			img.src = canvas.toDataURL();
			
			
		}

		// 마우스 이동에 따른 Crop 사각 박스을 이동하기 위한 이벤트 핸들러를 등록한다.
		function AddCropMoveEvent()
		{
			var canvas = document.getElementById("canvas");
			var bDrag = false;
			var iOldX, iOldY;

			canvas.onmousedown = function(e){
				// 화면 그리기 모드를 초기화하고 마우스 좌표 저장
				bDrag = true;
				iOldX = e.offsetX * (iImageWidth/rect.width);
				iOldY = e.offsetY * (iImageHeight/rect.height);
                iNewX = e.offsetX * (iImageWidth/rect.width); // 화면 기준 X
                iNewY = e.offsetY * (iImageHeight/rect.height); // 화면 기준 Y
			};

			canvas.onmousemove = function(e){
				if( bDrag == false ) return;

				// 새로운 좌표
				var iX = e.offsetX * (iImageWidth/rect.width);
				var iY = e.offsetY * (iImageHeight/rect.height);
				
				// 가로 좌표 저장
				iCropLeft = iOldX;
                iCropWidth = iX - iOldX;
                iNewX = iX;
				if( iCropLeft < 0 )
				{
					iCropLeft = 0;
				}
				else if( iCropLeft + iCropWidth > clsImage.width )
				{	
					iCropWidth = clsImage.width - iCropLeft;
				}

				iCropTop = iOldY;
                iCropHeight = iY - iOldY;
                iNewY = iY;
				if( iCropTop < 0 )
				{
					iCropTop = 0;
				}
				else if( iCropTop + iCropHeight > clsImage.height )
				{
					iCropHeight = clsImage.height - iCropTop;
				}

				// console.log("Origin", iCropLeft,iCropTop);
				// console.log("rect", iCropWidth, iCropHeight);
				DrawCropRect();
			};

			canvas.onmouseup = function(e){
				bDrag = false;
			};
		}

		function tossImage()
		{
			var canvas = document.getElementById("canvas_crop"),
				hidden = document.getElementById('tossed');

			img = new Image();
			img.onload = function(){
				var canvas = document.getElementById("tossed");
				canvas.width = iCropWidth;
				canvas.height = iCropHeight;
			};

			img.src = canvas.toDataURL();
			tossed.value = canvas.toDataURL();
			
		}

		