using UnityEngine;
using System.Collections;

public class SMAA : MonoBehaviour
{
	public bool ApplyEffect;
	public int State = 1;
	public int Passes = 1;

	private Texture2D black;
	private Shader shader;	
	private Material mat;

	

	/// <summary>
	/// 
	/// </summary>
	void Start()
	{
		shader = Shader.Find("Custom/SMAAshader");
		mat = new Material(shader);

		black = new Texture2D(1,1);
		black.SetPixel(0,0,new Color(0,0,0,0));
		black.Apply();

		//create texture generator
		GameObject obj = new GameObject();
		obj.name = "TextureGenerator";
		obj.AddComponent<AreaTexture>();
		obj.AddComponent<SearchTexture>();
	}


	/// <summary>
	/// 
	/// </summary>
	/// <param name="source"></param>
	/// <param name="destination"></param>
	void OnRenderImage(RenderTexture source, RenderTexture destination)
	{
		Graphics.Blit(black, destination);

		if (this.ApplyEffect)
		{
			if (State == 1)
			{
				Graphics.Blit(source, destination, mat, 0);
			}
			else if (State == 2)
			{
				mat.SetTexture("areaTex", GameObject.Find("TextureGenerator").GetComponent<AreaTexture>().alphaTex);
				mat.SetTexture("luminTex", GameObject.Find("TextureGenerator").GetComponent<AreaTexture>().luminTex);
				mat.SetTexture("searchTex", GameObject.Find("TextureGenerator").GetComponent<SearchTexture>().alphaTex);

				var rt = RenderTexture.GetTemporary(Screen.width, Screen.height, 0);

				Graphics.Blit(source, rt, mat, 0);
				Graphics.Blit(rt, destination, mat, 1);

				rt.Release();
			}
			else if (State == 3)
			{
				mat.SetTexture("areaTex", GameObject.Find("TextureGenerator").GetComponent<AreaTexture>().alphaTex);
				mat.SetTexture("luminTex", GameObject.Find("TextureGenerator").GetComponent<AreaTexture>().luminTex);
				mat.SetTexture("searchTex", GameObject.Find("TextureGenerator").GetComponent<SearchTexture>().alphaTex);
				mat.SetTexture("_SrcTex", source);

				var rt = RenderTexture.GetTemporary(Screen.width, Screen.height, 0);
				var rt2 = RenderTexture.GetTemporary(Screen.width, Screen.height, 0);
				var rt3 = RenderTexture.GetTemporary(Screen.width, Screen.height, 0);

				Graphics.Blit(source, rt3);
				for (var i = 0; i < Passes; i++)
				{
					Graphics.Blit(black, rt);
					Graphics.Blit(black, rt2);

					Graphics.Blit(rt3, rt, mat, 0);

					Graphics.Blit(rt, rt2, mat, 1);
					Graphics.Blit(rt2, rt3, mat, 2);
				}
				Graphics.Blit(rt3, destination);

				rt.Release();
				rt2.Release();
				rt3.Release();
			}		
		}
		else
		{
			Graphics.Blit(source, destination);
		}
	}
}
